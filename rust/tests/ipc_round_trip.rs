//! Integration test: IpcReceiver → channel → ViewerLoop end-to-end.

use std::time::Duration;

use scene_query_viewer::ipc::receiver::IpcReceiver;
use scene_query_viewer::viewer::overlay::ColorBuffer;
use scene_query_viewer::viewer::render_loop::ViewerLoop;
use tokio::sync::mpsc;

struct MockColorBuffer {
    highlighted: Vec<u32>,
}

impl MockColorBuffer {
    fn new() -> Self {
        Self {
            highlighted: Vec::new(),
        }
    }
}

impl ColorBuffer for MockColorBuffer {
    fn set_color(&mut self, primitive_id: u32, _color: [f32; 3]) {
        self.highlighted.push(primitive_id);
    }
    fn reset_colors(&mut self, _primitive_count: usize) {
        self.highlighted.clear();
    }
}

/// Send a length-prefixed MessagePack frame and read the Ack response.
async fn send_command(
    writer: &mut tokio::io::BufWriter<tokio::net::unix::OwnedWriteHalf>,
    reader: &mut tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
    payload: &[u8],
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let len = (payload.len() as u32).to_be_bytes();
    writer.write_all(&len).await.unwrap();
    writer.write_all(payload).await.unwrap();
    writer.flush().await.unwrap();

    // Read 4-byte response length
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await.unwrap();
    let resp_len = u32::from_be_bytes(len_buf) as usize;
    let mut resp = vec![0u8; resp_len];
    reader.read_exact(&mut resp).await.unwrap();

    let ack: std::collections::HashMap<String, String> =
        rmp_serde::from_slice(&resp).unwrap();
    assert_eq!(ack["status"], "ok");
}

#[tokio::test]
async fn test_highlight_round_trip() {
    let socket_path = "/tmp/scene-query-ipc-test.sock";
    let _ = std::fs::remove_file(socket_path);

    let (tx, rx) = mpsc::channel(16);
    let receiver = IpcReceiver::new(socket_path, tx);

    // Run IPC receiver in background
    tokio::spawn(async move {
        receiver.run().await.unwrap();
    });

    // Give the listener time to bind
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Connect and send a Highlight command
    let stream = tokio::net::UnixStream::connect(socket_path).await.unwrap();
    let (read_half, write_half) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(read_half);
    let mut writer = tokio::io::BufWriter::new(write_half);

    let highlight_cmd = rmp_serde::to_vec_named(&serde_json::json!({
        "op": "highlight",
        "primitive_ids": [0u32, 2u32, 4u32],
        "scores": [0.9f32, 0.7f32, 0.5f32],
        "color_map": "plasma"
    }))
    .unwrap();
    send_command(&mut writer, &mut reader, &highlight_cmd).await;

    // Give the channel a moment to deliver
    tokio::time::sleep(Duration::from_millis(10)).await;

    let mut vl = ViewerLoop::new(rx);
    let mut buf = MockColorBuffer::new();
    vl.frame(&mut buf, 8);

    assert_eq!(buf.highlighted.len(), 3, "three primitives should be highlighted");

    // Send a Clear command
    let stream2 = tokio::net::UnixStream::connect(socket_path).await.unwrap();
    let (read_half2, write_half2) = stream2.into_split();
    let mut reader2 = tokio::io::BufReader::new(read_half2);
    let mut writer2 = tokio::io::BufWriter::new(write_half2);

    let clear_cmd = rmp_serde::to_vec_named(&serde_json::json!({"op": "clear"})).unwrap();
    send_command(&mut writer2, &mut reader2, &clear_cmd).await;

    tokio::time::sleep(Duration::from_millis(10)).await;

    vl.frame(&mut buf, 8);
    assert_eq!(buf.highlighted.len(), 0, "highlights should be cleared");

    let _ = std::fs::remove_file(socket_path);
}
