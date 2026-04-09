//! Highlight overlay for query results in the threecrate viewer.

use std::collections::HashMap;
use std::str::FromStr;

/// Supported color maps for highlight rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMap {
    /// Plasma colormap (default): dark purple → orange → yellow.
    #[default]
    Plasma,
    /// Viridis colormap: dark purple → blue → green → yellow.
    Viridis,
    /// Hot colormap: black → red → yellow → white.
    Hot,
}

impl FromStr for ColorMap {
    type Err = std::convert::Infallible;

    /// Parse a color map name, defaulting to [`ColorMap::Plasma`] for unknown values.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "viridis" => Self::Viridis,
            "hot" => Self::Hot,
            _ => Self::Plasma,
        })
    }
}

/// Maps a score in [0, 1] to an RGB color using the specified colormap.
pub fn score_to_color(score: f32, color_map: ColorMap) -> [f32; 3] {
    let t = score.clamp(0.0, 1.0);
    match color_map {
        ColorMap::Plasma => {
            // Simplified plasma colormap approximation
            let r = (0.05 + 0.95 * t).min(1.0);
            let g = (0.5 * (1.0 - (2.0 * t - 1.0).powi(2))).max(0.0);
            let b = (0.5 * (1.0 - t)).max(0.0);
            [r, g, b]
        }
        ColorMap::Viridis => {
            // Simplified viridis: dark purple → blue-green → yellow
            let r = (0.267 + 0.726 * t * t).min(1.0);
            let g = (0.004 + 0.873 * t).min(1.0);
            let b = 0.329 * (1.0 - t) * (1.0 - t);
            [r, g, b]
        }
        ColorMap::Hot => {
            // Black → red → yellow → white (piecewise linear)
            let r = (3.0 * t).min(1.0);
            let g = (3.0 * t - 1.0).clamp(0.0, 1.0);
            let b = (3.0 * t - 2.0).clamp(0.0, 1.0);
            [r, g, b]
        }
    }
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
    pub fn apply(&mut self, primitive_ids: &[u32], scores: &[f32], color_map: &str) {
        self.highlights.clear();
        let map: ColorMap = color_map.parse().unwrap_or_default();
        for (&id, &score) in primitive_ids.iter().zip(scores.iter()) {
            self.highlights.insert(id, score_to_color(score, map));
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

    /// Iterate over all highlighted primitives and their colors.
    pub fn iter(&self) -> impl Iterator<Item = (u32, [f32; 3])> + '_ {
        self.highlights.iter().map(|(&id, &color)| (id, color))
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
        for map in [ColorMap::Plasma, ColorMap::Viridis, ColorMap::Hot] {
            let low = score_to_color(0.0, map);
            let high = score_to_color(1.0, map);
            for c in low.iter().chain(high.iter()) {
                assert!(
                    *c >= 0.0 && *c <= 1.0,
                    "color component out of bounds for {map:?}: {c}"
                );
            }
        }
    }

    #[test]
    fn test_color_map_from_str() {
        assert_eq!("plasma".parse::<ColorMap>().unwrap(), ColorMap::Plasma);
        assert_eq!("viridis".parse::<ColorMap>().unwrap(), ColorMap::Viridis);
        assert_eq!("hot".parse::<ColorMap>().unwrap(), ColorMap::Hot);
        assert_eq!("unknown".parse::<ColorMap>().unwrap(), ColorMap::Plasma);
    }

    #[test]
    fn test_apply_respects_color_map() {
        let mut overlay = HighlightOverlay::new();
        overlay.apply(&[0], &[1.0], "viridis");
        let viridis_color = overlay.color_for(0).unwrap();

        overlay.apply(&[0], &[1.0], "hot");
        let hot_color = overlay.color_for(0).unwrap();

        assert_ne!(viridis_color, hot_color, "different color maps should produce different colors");
    }
}
