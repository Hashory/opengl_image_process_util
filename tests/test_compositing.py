"""
Tests for the compositing module of opengl_image_process_util.
"""

import numpy as np
import pytest

from opengl_image_process_util.compositing import (
    _validate_images,
    add_blend,
    alpha_composite,
    blend_images,
    multiply_blend,
    overlay_images,
    screen_blend,
    subtract_blend,
)


class TestCompositing:
    """Tests for the image compositing functions."""

    def test_validate_images(self, rgba_image):
        """Test the image validation helper function."""
        # Should not raise for same shape RGBA images
        rgba_image2 = np.copy(rgba_image)
        _validate_images(rgba_image, rgba_image2)

        # Should raise for invalid images (not RGBA)
        rgb_image = rgba_image[:, :, :3]  # Remove alpha channel
        with pytest.raises(ValueError, match="must have 4 channels"):
            _validate_images(rgba_image, rgb_image)

        # Should raise for different size images
        with pytest.raises(ValueError, match="same dimensions"):
            _validate_images(rgba_image, rgba_image[:-10, :-10])

    def test_blend_images(self, rgba_image):
        """Test blending two images with alpha parameter."""
        # Create a uniform color RGBA image for easy testing
        second_img = np.ones_like(rgba_image) * 0.5
        second_img[:, :, 3] = 1.0  # Full opacity

        # Test default blending (50%)
        result = blend_images(rgba_image, second_img)
        assert result.shape == rgba_image.shape
        assert result.shape[2] == 4  # Should be RGBA

        # Test with alpha=0 (should be identical to first image)
        result_zero = blend_images(rgba_image, second_img, alpha=0.0)
        assert result_zero.shape == rgba_image.shape
        assert np.allclose(result_zero, rgba_image, atol=1e-5)

        # Test with alpha=1 (should be identical to second image)
        result_one = blend_images(rgba_image, second_img, alpha=1.0)
        assert result_one.shape == rgba_image.shape
        assert np.allclose(result_one, second_img, atol=1e-5)

        # Test with alpha=0.25
        result_quarter = blend_images(rgba_image, second_img, alpha=0.25)
        assert result_quarter.shape == rgba_image.shape
        # Verify correct blending with 0.25 alpha
        expected = rgba_image * 0.75 + second_img * 0.25
        assert np.allclose(result_quarter, expected, atol=1e-5)

    def test_overlay_images(self, rgba_image):
        """Test overlaying an RGBA image on top of another image."""
        # Create a test overlay image with varying alpha
        overlay_img = np.copy(rgba_image)
        h, w = rgba_image.shape[:2]

        # Create a gradient alpha from 0 to 1 horizontally
        for x in range(w):
            overlay_img[:, x, 3] = x / w

        # Test overlay
        result = overlay_images(rgba_image, overlay_img)
        assert result.shape == rgba_image.shape  # Should have alpha channel

        # For pixels where alpha is 0, should be the same as base rgba_image
        zero_alpha_mask = overlay_img[:, :, 3] == 0
        for i in range(3):  # Check RGB channels
            np.testing.assert_allclose(result[zero_alpha_mask, i], rgba_image[zero_alpha_mask, i], atol=1e-5)

        # For pixels where alpha is 1, should be the same as overlay_img
        one_alpha_mask = overlay_img[:, :, 3] == 1.0
        if np.any(one_alpha_mask):  # Only test if there are any pixels with alpha=1
            for i in range(3):  # Check RGB channels
                np.testing.assert_allclose(result[one_alpha_mask, i], overlay_img[one_alpha_mask, i], atol=1e-5)

    def test_alpha_composite(self, rgba_image):
        """Test alpha compositing."""
        # Create a test foreground image with varying alpha
        foreground = np.copy(rgba_image)
        foreground[:, :, 0] = 1.0  # Red channel
        foreground[:, :, 1:3] = 0.0  # Green and Blue channels
        h, w = rgba_image.shape[:2]

        # Create a gradient alpha from 0 to 1 horizontally
        for x in range(w):
            foreground[:, x, 3] = x / w

        # Alpha composite should be the same as overlay_images
        result1 = alpha_composite(rgba_image, foreground)
        result2 = overlay_images(rgba_image, foreground)
        assert np.allclose(result1, result2, atol=1e-5)

    def test_multiply_blend(self, rgba_image):
        """Test multiply blend mode."""
        # Create a uniform color image for easy testing
        second_img = np.ones_like(rgba_image) * 0.5
        second_img[:, :, 3] = 1.0  # Full opacity

        result = multiply_blend(rgba_image, second_img)
        assert result.shape == rgba_image.shape

        # Multiply blend should multiply each RGB channel but preserve alpha
        expected = np.copy(rgba_image)
        expected[:, :, :3] = rgba_image[:, :, :3] * second_img[:, :, :3]
        assert np.allclose(result, expected, atol=1e-5)

    def test_screen_blend(self, rgba_image):
        """Test screen blend mode."""
        # Create a uniform color image for easy testing
        second_img = np.ones_like(rgba_image) * 0.5
        second_img[:, :, 3] = 1.0  # Full opacity

        result = screen_blend(rgba_image, second_img)
        assert result.shape == rgba_image.shape

        # Screen blend formula: 1 - (1-a) * (1-b) for RGB channels only
        expected = np.copy(rgba_image)
        expected[:, :, :3] = 1.0 - (1.0 - rgba_image[:, :, :3]) * (1.0 - second_img[:, :, :3])
        assert np.allclose(result, expected, atol=1e-5)

    def test_add_blend(self, rgba_image):
        """Test add blend mode."""
        # Create a uniform color image for easy testing
        second_img = np.ones_like(rgba_image) * 0.2
        second_img[:, :, 3] = 1.0  # Full opacity

        result = add_blend(rgba_image, second_img)
        assert result.shape == rgba_image.shape

        # Add blend should add RGB values and clamp, but preserve alpha
        expected = np.copy(rgba_image)
        expected[:, :, :3] = np.clip(rgba_image[:, :, :3] + second_img[:, :, :3], 0.0, 1.0)
        assert np.allclose(result, expected, atol=1e-5)

        # Test with values that would exceed 1.0
        bright_img = np.ones_like(rgba_image) * 0.8
        bright_img[:, :, 3] = 1.0  # Full opacity
        result_exceeded = add_blend(rgba_image, bright_img)
        assert np.max(result_exceeded[:, :, :3]) <= 1.0  # RGB should be clamped
        assert np.allclose(result_exceeded[:, :, 3], rgba_image[:, :, 3], atol=1e-5)  # Alpha preserved

    def test_subtract_blend(self, rgba_image):
        """Test subtract blend mode."""
        # Create a uniform color image for easy testing
        second_img = np.ones_like(rgba_image) * 0.2
        second_img[:, :, 3] = 1.0  # Full opacity

        result = subtract_blend(rgba_image, second_img)
        assert result.shape == rgba_image.shape

        # Subtract blend should subtract RGB values and clamp, but preserve alpha
        expected = np.copy(rgba_image)
        expected[:, :, :3] = np.clip(rgba_image[:, :, :3] - second_img[:, :, :3], 0.0, 1.0)
        assert np.allclose(result, expected, atol=1e-5)

        # Test with values that would go below 0.0
        dark_img = np.zeros_like(rgba_image)
        dark_img[:, :, :3] = 0.8  # High values to subtract
        dark_img[:, :, 3] = 1.0  # Full opacity

        # Using a fixed value for the base image to ensure predictable results
        base_img = np.zeros_like(rgba_image)
        base_img[:, :, :3] = 0.3  # Low values
        base_img[:, :, 3] = 1.0  # Full opacity

        result_below = subtract_blend(base_img, dark_img)
        assert np.min(result_below[:, :, :3]) >= 0.0  # RGB should be clamped
        assert np.allclose(result_below[:, :, 3], base_img[:, :, 3], atol=1e-5)  # Alpha preserved
