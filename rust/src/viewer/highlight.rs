//! Highlight overlay for query results in the threecrate viewer.

use std::collections::HashMap;

/// Maps a score in [0, 1] to an RGB color using a Plasma-like colormap.
pub fn score_to_color(score: f32) -> [f32; 3] {
    // Simplified plasma colormap approximation
    let t = score.clamp(0.0, 1.0);
    let r = (0.05 + 0.95 * t).min(1.0);
    let g = (0.0 + 0.5 * (1.0 - (2.0 * t - 1.0).powi(2))).max(0.0);
    let b = (0.5 * (1.0 - t)).max(0.0);
    [r, g, b]
}

/// Holds the current highlight state for the viewer.
#[derive(Debug, Default)]
pub struct HighlightOverlay {
    /// Maps primitive_id → highlight color [R, G, B] in [0, 1].
    highlights: HashMap<u32, [f32; 3]>,
}

impl HighlightOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply highlight command: set colors for the given primitives.
    pub fn apply(&mut self, primitive_ids: &[u32], scores: &[f32], _color_map: &str) {
        self.highlights.clear();
        for (&id, &score) in primitive_ids.iter().zip(scores.iter()) {
            self.highlights.insert(id, score_to_color(score));
        }
    }

    /// Clear all highlights.
    pub fn clear(&mut self) {
        self.highlights.clear();
    }

    /// Return the highlight color for a primitive, or None if not highlighted.
    pub fn color_for(&self, primitive_id: u32) -> Option<[f32; 3]> {
        self.highlights.get(&primitive_id).copied()
    }

    pub fn highlighted_count(&self) -> usize {
        self.highlights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_and_clear() {
        let mut overlay = HighlightOverlay::new();
        overlay.apply(&[0, 1, 2], &[1.0, 0.8, 0.5], "plasma");
        assert_eq!(overlay.highlighted_count(), 3);
        assert!(overlay.color_for(0).is_some());
        assert!(overlay.color_for(99).is_none());

        overlay.clear();
        assert_eq!(overlay.highlighted_count(), 0);
    }

    #[test]
    fn test_score_to_color_bounds() {
        let low = score_to_color(0.0);
        let high = score_to_color(1.0);
        for c in low.iter().chain(high.iter()) {
            assert!(*c >= 0.0 && *c <= 1.0, "color component out of bounds: {c}");
        }
    }
}
