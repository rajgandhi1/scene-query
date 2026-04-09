//! Query result overlay rendering (threecrate integration).
//!
//! Provides the [`ColorBuffer`] trait that threecrate's `PointRenderer` /
//! `SplatRenderer` should implement, and wires [`HighlightOverlay`] colors
//! into the render loop via [`QueryOverlay::render_frame`].

use crate::viewer::highlight::HighlightOverlay;

/// Abstraction over a GPU color buffer.
///
/// Implement this on threecrate's `PointRenderer` or `SplatRenderer` to feed
/// highlight colors into the real viewer.
pub trait ColorBuffer {
    /// Set the highlight color for a primitive.
    fn set_color(&mut self, primitive_id: u32, color: [f32; 3]);
    /// Reset all primitives to their default (un-highlighted) color.
    fn reset_colors(&mut self, primitive_count: usize);
}

/// Overlay renderer — wraps HighlightOverlay and applies it to the color buffer.
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
    /// Resets the buffer to default colors, then writes the highlight color
    /// for each currently highlighted primitive.
    pub fn render_frame(&self, buf: &mut impl ColorBuffer, primitive_count: usize) {
        buf.reset_colors(primitive_count);
        for (id, color) in self.highlights.iter() {
            buf.set_color(id, color);
        }
    }
}

impl Default for QueryOverlay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockColorBuffer {
        colors: Vec<Option<[f32; 3]>>,
        reset_count: usize,
    }

    impl MockColorBuffer {
        fn new(size: usize) -> Self {
            Self {
                colors: vec![None; size],
                reset_count: 0,
            }
        }
    }

    impl ColorBuffer for MockColorBuffer {
        fn set_color(&mut self, primitive_id: u32, color: [f32; 3]) {
            if let Some(slot) = self.colors.get_mut(primitive_id as usize) {
                *slot = Some(color);
            }
        }

        fn reset_colors(&mut self, _primitive_count: usize) {
            self.colors.iter_mut().for_each(|c| *c = None);
            self.reset_count += 1;
        }
    }

    #[test]
    fn test_render_frame_applies_highlights() {
        let mut overlay = QueryOverlay::new();
        overlay.highlights.apply(&[0, 2], &[1.0, 0.5], "plasma");

        let mut buf = MockColorBuffer::new(4);
        overlay.render_frame(&mut buf, 4);

        assert_eq!(buf.reset_count, 1, "reset should be called once per frame");
        assert!(buf.colors[0].is_some(), "primitive 0 should be highlighted");
        assert!(buf.colors[1].is_none(), "primitive 1 should not be highlighted");
        assert!(buf.colors[2].is_some(), "primitive 2 should be highlighted");
        assert!(buf.colors[3].is_none(), "primitive 3 should not be highlighted");
    }

    #[test]
    fn test_render_frame_after_clear() {
        let mut overlay = QueryOverlay::new();
        overlay.highlights.apply(&[0], &[1.0], "plasma");

        let mut buf = MockColorBuffer::new(2);
        overlay.render_frame(&mut buf, 2);
        assert!(buf.colors[0].is_some());

        overlay.highlights.clear();
        overlay.render_frame(&mut buf, 2);
        assert!(buf.colors[0].is_none(), "cleared overlay should produce no highlights");
        assert_eq!(buf.reset_count, 2);
    }
}
