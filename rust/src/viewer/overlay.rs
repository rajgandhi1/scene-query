//! Query result overlay rendering (threecrate integration stub).
//!
//! Phase 2: wire this into the threecrate render loop to display highlighted
//! primitives as colored points/splats in the interactive viewer.

use crate::viewer::highlight::HighlightOverlay;

/// Overlay renderer — wraps HighlightOverlay and provides a rendering hook.
pub struct QueryOverlay {
    pub highlights: HighlightOverlay,
}

impl QueryOverlay {
    pub fn new() -> Self {
        Self {
            highlights: HighlightOverlay::new(),
        }
    }

    /// Called once per frame by the threecrate render loop.
    ///
    /// For each highlighted primitive, this should modify the vertex color
    /// buffer before the draw call. Stubbed until threecrate integration in Phase 2.
    pub fn render_frame(&self, _primitive_count: usize) {
        // TODO Phase 2: iterate highlighted primitives, update GPU color buffer
        // via threecrate's PointRenderer or SplatRenderer API.
    }
}

impl Default for QueryOverlay {
    fn default() -> Self {
        Self::new()
    }
}
