//! scene-query viewer bridge
//!
//! Rust side of the Python↔Rust IPC bridge. Receives highlight commands
//! from the Python FastAPI layer over a Unix socket and applies them to
//! the threecrate 3D viewer.

pub mod ipc;
pub mod viewer;

pub use ipc::receiver::IpcReceiver;
pub use viewer::highlight::{ColorMap, HighlightOverlay};
pub use viewer::overlay::{ColorBuffer, QueryOverlay};
pub use viewer::render_loop::ViewerLoop;
pub use viewer::threecrate_buffer::{PointVertexBuffer, DEFAULT_COLOR};
