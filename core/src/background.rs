//! Removing the still background from video frames before PIV.
//!
//! PIV finds the shift that best lines up the *texture* of two frames. In a
//! lab, the only texture is the tracer particles. In a real video there is
//! usually more: the bottom of a stream seen through clear water, the walls
//! of a tank, a printed pattern, reflections of the sky. That texture does
//! not move, so it pulls every vector towards zero, and in a phone video of
//! a shallow canal it dominated so strongly that ~70 % of the vectors on the
//! water read "still".
//!
//! The standard fix (Raffel et al., *Particle Image Velocimetry*, §5.4) is
//! to subtract a background image. With a video we can estimate the
//! background from the video itself: the **per-pixel median** of frames
//! spread over a second or two. Something still sits at the same pixel in
//! every frame, so it is the median. A moving particle visits a given pixel
//! in only a few frames, so the median ignores it. `|frame − background|`
//! then keeps the particles and removes the still scene. The absolute value
//! means bright particles on a dark background and dark particles (pepper)
//! on a light one both work.
//!
//! Limits: anything that barely moves during the sampled frames counts as
//! background and is removed too, so for very slow flows the frames must
//! span a longer time. The camera must not move (use a tripod), otherwise
//! the "still" scene smears and is only partly removed.

use crate::image::Image;

/// Reasons the background can't be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundError {
    /// Fewer than 3 frames: a median of 1 or 2 values can't reject anything.
    TooFewFrames,
    /// The frames are not all the same size.
    SizeMismatch,
}

impl std::fmt::Display for BackgroundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BackgroundError::TooFewFrames => "need at least 3 frames to estimate a background",
            BackgroundError::SizeMismatch => "background frames have different sizes",
        })
    }
}

impl std::error::Error for BackgroundError {}

/// Per-pixel median of `frames`.
///
/// With an even count the lower middle value is used, which is fine here and
/// avoids averaging a particle into the background.
pub fn median_background(frames: &[&Image]) -> Result<Image, BackgroundError> {
    if frames.len() < 3 {
        return Err(BackgroundError::TooFewFrames);
    }
    let (w, h) = (frames[0].width(), frames[0].height());
    if frames.iter().any(|f| f.width() != w || f.height() != h) {
        return Err(BackgroundError::SizeMismatch);
    }
    let mut out = Image::new(w, h);
    // One small buffer reused for every pixel.
    let mut values = Vec::with_capacity(frames.len());
    for i in 0..w * h {
        values.clear();
        values.extend(frames.iter().map(|f| f.data()[i]));
        // `select_nth_unstable_by` puts the middle element in place without
        // sorting everything: O(n) instead of O(n log n) per pixel.
        let mid = (values.len() - 1) / 2;
        values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        out.set(i % w, i / w, values[mid]);
    }
    Ok(out)
}

/// `|img − background|` per pixel: the still scene removed, moving
/// particles (bright or dark) kept as bright spots.
///
/// Panics if the sizes differ (a programming error, not bad input:
/// callers build both from the same frames).
pub fn subtract_background(img: &Image, background: &Image) -> Image {
    assert_eq!(
        (img.width(), img.height()),
        (background.width(), background.height()),
        "image and background must be the same size"
    );
    let mut out = Image::new(img.width(), img.height());
    for (i, (p, b)) in img.data().iter().zip(background.data()).enumerate() {
        out.set(i % img.width(), i / img.width(), (p - b).abs());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, f(x, y));
            }
        }
        img
    }

    #[test]
    fn median_ignores_a_particle_that_visits_one_frame() {
        let still = |x: usize, y: usize| ((x * 7 + y * 13) % 10) as f32 / 10.0;
        let mut frames: Vec<Image> = (0..5).map(|_| filled(8, 8, still)).collect();
        frames[2].set(3, 4, 1.0); // a particle passes through in one frame
        let refs: Vec<&Image> = frames.iter().collect();
        let bg = median_background(&refs).unwrap();
        assert_eq!(bg, filled(8, 8, still));
    }

    #[test]
    fn subtraction_keeps_bright_and_dark_particles() {
        let bg = filled(4, 4, |_, _| 0.5);
        let mut img = bg.clone();
        img.set(1, 1, 0.9); // bright particle
        img.set(2, 3, 0.1); // dark particle (pepper on a light dish)
        let d = subtract_background(&img, &bg);
        assert!((d.get(1, 1) - 0.4).abs() < 1e-6 && (d.get(2, 3) - 0.4).abs() < 1e-6);
        assert_eq!(d.get(0, 0), 0.0); // still scene removed
    }

    #[test]
    fn too_few_or_mismatched_frames_are_rejected() {
        let a = Image::new(4, 4);
        let b = Image::new(5, 4);
        assert_eq!(
            median_background(&[&a, &a]),
            Err(BackgroundError::TooFewFrames)
        );
        assert_eq!(
            median_background(&[&a, &a, &b]),
            Err(BackgroundError::SizeMismatch)
        );
    }
}
