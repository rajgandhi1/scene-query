//! [`ColorBuffer`] implementation backed by threecrate's [`PointVertex`] slice.
//!
//! # Usage
//!
//! ```ignore
//! // Build your vertex slice from the loaded point cloud (once)
//! let mut vertices = threecrate_gpu::renderer::point_cloud_to_vertices(
//!     &point_cloud, DEFAULT_COLOR, 1.0,
//! );
//!
//! // In your winit frame loop:
//! let mut buf = PointVertexBuffer::new(&mut vertices);
//! viewer_loop.frame(&mut buf, vertices.len());
//! renderer.render(&vertices).unwrap();
//! ```

use threecrate_gpu::renderer::PointVertex;

use crate::viewer::overlay::ColorBuffer;

/// Un-highlighted point color (white). Override with [`PointVertexBuffer::with_default_color`].
pub const DEFAULT_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// [`ColorBuffer`] that writes highlight colors directly into a [`PointVertex`] slice.
///
/// Pass to [`crate::viewer::render_loop::ViewerLoop::frame`], then call
/// [`threecrate_gpu::renderer::PointCloudRenderer::render`] with the same slice.
pub struct PointVertexBuffer<'a> {
    vertices: &'a mut Vec<PointVertex>,
    default_color: [f32; 3],
}

impl<'a> PointVertexBuffer<'a> {
    pub fn new(vertices: &'a mut Vec<PointVertex>) -> Self {
        Self {
            vertices,
            default_color: DEFAULT_COLOR,
        }
    }

    /// Override the default (un-highlighted) color.
    pub fn with_default_color(mut self, color: [f32; 3]) -> Self {
        self.default_color = color;
        self
    }
}

impl ColorBuffer for PointVertexBuffer<'_> {
    fn set_color(&mut self, primitive_id: u32, color: [f32; 3]) {
        if let Some(v) = self.vertices.get_mut(primitive_id as usize) {
            v.color = color;
        }
    }

    fn reset_colors(&mut self, primitive_count: usize) {
        let count = primitive_count.min(self.vertices.len());
        for v in &mut self.vertices[..count] {
            v.color = self.default_color;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vertex() -> PointVertex {
        PointVertex {
            position: [0.0; 3],
            color: DEFAULT_COLOR,
            size: 1.0,
            normal: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn test_set_color_updates_vertex() {
        let mut vertices: Vec<PointVertex> = (0..4).map(|_| make_vertex()).collect();
        let mut buf = PointVertexBuffer::new(&mut vertices);

        buf.set_color(1, [1.0, 0.0, 0.0]);
        buf.set_color(3, [0.0, 1.0, 0.0]);

        assert_eq!(buf.vertices[0].color, DEFAULT_COLOR);
        assert_eq!(buf.vertices[1].color, [1.0, 0.0, 0.0]);
        assert_eq!(buf.vertices[2].color, DEFAULT_COLOR);
        assert_eq!(buf.vertices[3].color, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_reset_restores_default_color() {
        let mut vertices: Vec<PointVertex> = (0..4).map(|_| make_vertex()).collect();
        let mut buf = PointVertexBuffer::new(&mut vertices);

        buf.set_color(0, [1.0, 0.5, 0.0]);
        buf.set_color(2, [0.0, 0.5, 1.0]);
        buf.reset_colors(4);

        for v in &*buf.vertices {
            assert_eq!(v.color, DEFAULT_COLOR);
        }
    }

    #[test]
    fn test_out_of_bounds_set_is_ignored() {
        let mut vertices: Vec<PointVertex> = vec![make_vertex()];
        let mut buf = PointVertexBuffer::new(&mut vertices);
        buf.set_color(99, [1.0, 0.0, 0.0]); // should not panic
        assert_eq!(buf.vertices[0].color, DEFAULT_COLOR);
    }

    #[test]
    fn test_custom_default_color() {
        let gray = [0.5, 0.5, 0.5];
        let mut vertices: Vec<PointVertex> = vec![make_vertex()];
        let mut buf = PointVertexBuffer::new(&mut vertices).with_default_color(gray);
        buf.set_color(0, [1.0, 0.0, 0.0]);
        buf.reset_colors(1);
        assert_eq!(buf.vertices[0].color, gray);
    }
}
