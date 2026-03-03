//! Unix socket IPC receiver — listens for highlight commands from Python.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub const DEFAULT_SOCKET_PATH: &str = "/tmp/scene-query-viewer.sock";

#[derive(Debug, Error)]
pub enum IpcError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Deserialization error: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("Serialization error: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
}

/// Commands sent from Python to the Rust viewer.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ViewerCommand {
    Highlight {
        primitive_ids: Vec<u32>,
        scores: Vec<f32>,
        color_map: String,
    },
    Clear,
}

/// Acknowledgement sent back to Python.
#[derive(Debug, Serialize)]
pub struct Ack {
    pub status: &'static str,
}

impl Ack {
    pub fn ok() -> Self {
        Self { status: "ok" }
    }
    pub fn err() -> Self {
        Self { status: "error" }
    }
}

/// Listens on a Unix socket and forwards decoded commands over an mpsc channel.
pub struct IpcReceiver {
    socket_path: PathBuf,
    tx: mpsc::Sender<ViewerCommand>,
}

impl IpcReceiver {
    pub fn new(socket_path: impl AsRef<Path>, tx: mpsc::Sender<ViewerCommand>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
            tx,
        }
    }

    /// Start listening. Spawns a Tokio task per incoming connection.
    pub async fn run(self) -> Result<(), IpcError> {
        // Remove stale socket file if present
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        let listener = UnixListener::bind(&self.socket_path)?;
        info!("IPC receiver listening on {:?}", self.socket_path);

        let tx = Arc::new(self.tx);
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let tx = Arc::clone(&tx);
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, tx).await {
                            warn!("IPC connection error: {e}");
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {e}");
                }
            }
        }
    }
}

async fn handle_connection(
    mut stream: UnixStream,
    tx: Arc<mpsc::Sender<ViewerCommand>>,
) -> Result<(), IpcError> {
    info!("New IPC connection");

    loop {
        // Read 4-byte length prefix
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                info!("IPC client disconnected");
                return Ok(());
            }
            Err(e) => return Err(IpcError::Io(e)),
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; msg_len];
        stream.read_exact(&mut buf).await?;

        let cmd: ViewerCommand = rmp_serde::from_slice(&buf)?;

        let ack = if tx.send(cmd).await.is_ok() {
            Ack::ok()
        } else {
            warn!("Command channel closed");
            Ack::err()
        };

        let response = rmp_serde::to_vec_named(&ack)?;
        let resp_len = (response.len() as u32).to_be_bytes();
        stream.write_all(&resp_len).await?;
        stream.write_all(&response).await?;
    }
}
