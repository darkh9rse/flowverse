//! A minimal grayscale image.

use std::fmt;

/// A grayscale image stored row by row ("row-major").
///
/// Pixel `(x, y)` lives at index `y * width + x` in `data`. `x` grows to the
/// right and `y` grows downwards.
///
/// Brightness is an `f32`. No range is enforced. By convention, 0.0 is black
/// and 1.0 is the peak brightness of one synthetic particle. `f32` rather
/// than `u8` because PIV subtracts means and multiplies values, which needs
/// fractions and negatives. `f32` rather than `f64` because a video frame has
/// millions of pixels and half the memory matters, especially in the browser.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    // The fields are private (no `pub`) so nobody can create an Image whose
    // `data` length disagrees with `width * height`. All construction goes
    // through `new` / `from_vec`, which check this.
    width: usize,
    height: usize,
    data: Vec<f32>,
}

/// Error returned when raw pixel data does not match the stated size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeMismatch {
    pub expected: usize,
    pub actual: usize,
}

// Implementing `Display` lets the error be printed with `{}`;
// implementing `std::error::Error` lets it work with `?` and `Box<dyn Error>`.
impl fmt::Display for SizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pixel buffer has {} values but width*height is {}",
            self.actual, self.expected
        )
    }
}

impl std::error::Error for SizeMismatch {}

impl Image {
    /// A black image (all pixels 0.0).
    pub fn new(width: usize, height: usize) -> Self {
        Image {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// Wrap an existing row-major pixel buffer.
    ///
    /// Returns an error instead of panicking if the length is wrong, because
    /// this is the function that receives data from outside (e.g. a video
    /// frame from JavaScript).
    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Result<Self, SizeMismatch> {
        if data.len() != width * height {
            return Err(SizeMismatch {
                expected: width * height,
                actual: data.len(),
            });
        }
        Ok(Image {
            width,
            height,
            data,
        })
    }

    /// Convert 8-bit RGBA pixels (4 bytes per pixel, as in a browser
    /// `ImageData`) to grayscale brightness 0.0–1.0.
    ///
    /// Uses the Rec. 601 luma weights `0.299 R + 0.587 G + 0.114 B`, which
    /// roughly match how bright each colour looks to the eye. For PIV the
    /// exact weights hardly matter. Alpha is ignored.
    pub fn from_rgba8(width: usize, height: usize, rgba: &[u8]) -> Result<Self, SizeMismatch> {
        if rgba.len() != width * height * 4 {
            return Err(SizeMismatch {
                expected: width * height * 4,
                actual: rgba.len(),
            });
        }
        // `chunks_exact(4)` walks the bytes one pixel (r, g, b, a) at a time.
        let data = rgba
            .chunks_exact(4)
            .map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
            .collect();
        Ok(Image {
            width,
            height,
            data,
        })
    }

    /// The inverse, for display and tests: gray RGBA bytes, opaque.
    /// Values are clamped to 0.0–1.0 first.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for &p in &self.data {
            let g = (p.clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[g, g, g, 255]);
        }
        out
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Brightness at `(x, y)`. Panics if out of bounds, just like indexing a
    /// slice with `v[i]`. Out-of-bounds here is a programming bug, not bad
    /// user input, so a loud panic is the right response.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.index(x, y)]
    }

    /// Set the brightness at `(x, y)`. Panics if out of bounds.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        let i = self.index(x, y);
        self.data[i] = value;
    }

    /// Add to the brightness at `(x, y)`. Overlapping particles add up,
    /// like light does on a camera sensor.
    pub fn add(&mut self, x: usize, y: usize, value: f32) {
        let i = self.index(x, y);
        self.data[i] += value;
    }

    /// All pixels as one row-major slice (read-only).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    fn index(&self, x: usize, y: usize) -> usize {
        // Checking x and y separately matters: x = width, y = 0 would give a
        // valid-looking index (the first pixel of row 1) and silently read
        // the wrong pixel.
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) is outside a {}x{} image",
            self.width,
            self.height
        );
        y * self.width + x
    }
}

// `#[cfg(test)]` means: only compile this module when running `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_image_is_black_and_row_major() {
        let mut img = Image::new(4, 3);
        assert!(img.data().iter().all(|&p| p == 0.0));
        img.set(1, 2, 5.0);
        // Row 2, column 1 → index 2*4 + 1 = 9.
        assert_eq!(img.data()[9], 5.0);
        img.add(1, 2, 1.5);
        assert_eq!(img.get(1, 2), 6.5);
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        assert!(Image::from_vec(2, 2, vec![0.0; 4]).is_ok());
        let err = Image::from_vec(2, 2, vec![0.0; 3]).unwrap_err();
        assert_eq!(
            err,
            SizeMismatch {
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn rgba_conversion_uses_luma_weights_and_checks_length() {
        let rgba = [255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 7];
        let img = Image::from_rgba8(3, 1, &rgba).unwrap();
        assert!((img.get(0, 0) - 0.299).abs() < 1e-6);
        assert!((img.get(1, 0) - 0.587).abs() < 1e-6);
        assert!((img.get(2, 0) - 1.0).abs() < 1e-6);
        assert!(Image::from_rgba8(2, 1, &rgba).is_err());
        // Round trip through to_rgba8: white stays white.
        assert_eq!(&img.to_rgba8()[8..12], &[255, 255, 255, 255]);
    }

    #[test]
    #[should_panic]
    fn x_past_the_edge_panics_instead_of_wrapping() {
        Image::new(4, 3).get(4, 0);
    }
}
