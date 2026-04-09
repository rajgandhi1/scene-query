//! Viewer render loop: dispatches IPC commands to [`QueryOverlay`] each frame.

use tokio::sync::mpsc;

use crate::ipc::receiver::ViewerCommand;
use crate::viewer::overlay::{ColorBuffer, QueryOverlay};

/// Connects the IPC command channel to the highlight overlay, driving
/// highlight updates each frame.
///
/// Typical usage:
/// ```ignore
/// let (tx, rx) = mpsc::channel(64);
/// tokio::spawn(IpcReceiver::new(socket_path, tx).run());
/// let mut viewer_loop = ViewerLoop::new(rx);
///
/// // inside your threecrate frame callback:
/// viewer_loop.frame(&mut point_renderer, scene.primitive_count());
/// ```
pub struct ViewerLoop {
    overlay: QueryOverlay,
    rx: mpsc::Receiver<ViewerCommand>,
}

impl ViewerLoop {
    pub fn new(rx: mpsc::Receiver<ViewerCommand>) -> Self {
        Self {
            overlay: QueryOverlay::new(),
            rx,
        }
    }

    /// Drain all pending IPC commands without blocking.
    pub fn process_commands(&mut self) {
        while let Ok(cmd) = self.rx.try_recv() {
            match cmd {
                ViewerCommand::Highlight {
                    primitive_ids,
                    scores,
                    color_map,
                } => {
                    self.overlay
                        .highlights
                        .apply(&primitive_ids, &scores, &color_map);
                }
                ViewerCommand::Clear => {
                    self.overlay.highlights.clear();
                }
            }
        }
    }

    /// Call once per frame: consume pending IPC commands, then render highlights.
    pub fn frame(&mut self, buf: &mut impl ColorBuffer, primitive_count: usize) {
        self.process_commands();
        self.overlay.render_frame(buf, primitive_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_highlight_command_applied_on_next_frame() {
        let (tx, rx) = mpsc::channel(16);
        let mut vl = ViewerLoop::new(rx);
        let mut buf = MockColorBuffer::new();

        tx.try_send(ViewerCommand::Highlight {
            primitive_ids: vec![1, 3, 5],
            scores: vec![0.9, 0.7, 0.5],
            color_map: "plasma".to_string(),
        })
        .unwrap();

        vl.frame(&mut buf, 8);
        assert_eq!(buf.highlighted.len(), 3);
    }

    #[test]
    fn test_clear_command_removes_highlights() {
        let (tx, rx) = mpsc::channel(16);
        let mut vl = ViewerLoop::new(rx);
        let mut buf = MockColorBuffer::new();

        tx.try_send(ViewerCommand::Highlight {
            primitive_ids: vec![0, 1],
            scores: vec![1.0, 0.8],
            color_map: "viridis".to_string(),
        })
        .unwrap();
        vl.frame(&mut buf, 4);
        assert_eq!(buf.highlighted.len(), 2);

        tx.try_send(ViewerCommand::Clear).unwrap();
        vl.frame(&mut buf, 4);
        assert_eq!(buf.highlighted.len(), 0, "highlights should be cleared after Clear command");
    }

    #[test]
    fn test_multiple_color_maps() {
        let (tx, rx) = mpsc::channel(16);
        let mut vl = ViewerLoop::new(rx);
        let mut buf = MockColorBuffer::new();

        for map in ["plasma", "viridis", "hot"] {
            tx.try_send(ViewerCommand::Highlight {
                primitive_ids: vec![0],
                scores: vec![0.5],
                color_map: map.to_string(),
            })
            .unwrap();
            vl.frame(&mut buf, 2);
            assert_eq!(buf.highlighted.len(), 1, "should highlight with colormap: {map}");
        }
    }

    #[test]
    fn test_no_commands_no_highlights() {
        let (_tx, rx) = mpsc::channel(16);
        let mut vl = ViewerLoop::new(rx);
        let mut buf = MockColorBuffer::new();

        vl.frame(&mut buf, 4);
        assert_eq!(buf.highlighted.len(), 0);
    }
}
