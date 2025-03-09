"""
Tests for the effects module of opengl_image_process_util.
"""

import numpy as np

from opengl_image_process_util.core import validate_rgba
from opengl_image_process_util.effects import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    blur_image,
    edge_detection,
    grayscale,
    invert_colors,
    sharpen_image,
    vignette,
)


class TestEffects:
    """Tests for the image effects functions."""

    def test_blur_image(self, rgba_image):
        """Test Gaussian blur effect."""
        # Test with default parameters
        result = blur_image(rgba_image)

        # Validate output is RGBA
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Test with different radius
        result_stronger = blur_image(rgba_image, radius=5.0)
        validate_rgba(result_stronger)
        assert result_stronger.shape == rgba_image.shape

        # Check that alpha channel is preserved
        assert np.allclose(result[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

    def test_sharpen_image(self, rgba_image):
        """Test image sharpening effect."""
        # Test with default parameters
        result = sharpen_image(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Test with different strength
        result_stronger = sharpen_image(rgba_image, strength=2.0)
        validate_rgba(result_stronger)
        assert result_stronger.shape == rgba_image.shape

        # Check that alpha channel is preserved
        assert np.allclose(result[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

    def test_adjust_brightness(self, rgba_image):
        """Test brightness adjustment."""
        # Test with default parameters (no change)
        result = adjust_brightness(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape
        assert np.allclose(result, rgba_image, atol=1e-5)

        # Test darkening
        result_darker = adjust_brightness(rgba_image, factor=0.5)
        validate_rgba(result_darker)
        assert result_darker.shape == rgba_image.shape

        # Should be darker than original (only RGB channels)
        assert np.mean(result_darker[:, :, :3]) < np.mean(rgba_image[:, :, :3])

        # Check that alpha channel is preserved
        assert np.allclose(result_darker[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

        # Test brightening
        result_brighter = adjust_brightness(rgba_image, factor=1.5)
        validate_rgba(result_brighter)
        assert result_brighter.shape == rgba_image.shape

        # Should be brighter than original (only RGB channels)
        assert np.mean(result_brighter[:, :, :3]) > np.mean(rgba_image[:, :, :3])

        # Check that alpha channel is preserved
        assert np.allclose(result_brighter[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

        # Expected results verification
        expected_darker = np.copy(rgba_image)
        expected_darker[:, :, :3] = rgba_image[:, :, :3] * 0.5
        assert np.allclose(result_darker, expected_darker, atol=1e-5)

    def test_adjust_contrast(self, rgba_image):
        """Test contrast adjustment."""
        # Test with default parameters (no change)
        result = adjust_contrast(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Test lowering contrast
        result_lower = adjust_contrast(rgba_image, factor=0.5)
        validate_rgba(result_lower)
        assert result_lower.shape == rgba_image.shape

        # Test increasing contrast
        result_higher = adjust_contrast(rgba_image, factor=1.5)
        validate_rgba(result_higher)
        assert result_higher.shape == rgba_image.shape

        # Verify that increasing contrast increases standard deviation (RGB channels only)
        std_orig = np.std(rgba_image[:, :, :3])
        std_higher = np.std(result_higher[:, :, :3])
        std_lower = np.std(result_lower[:, :, :3])
        assert std_higher > std_orig
        assert std_lower < std_orig

        # Check that alpha channel is preserved
        assert np.allclose(result_lower[:, :, 3], rgba_image[:, :, 3], atol=1e-5)
        assert np.allclose(result_higher[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

    def test_adjust_saturation(self, rgba_image):
        """Test saturation adjustment."""
        # Test with default parameters (no change)
        result = adjust_saturation(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape
        assert np.allclose(result, rgba_image, atol=1e-5)

        # Test lowering saturation
        result_lower = adjust_saturation(rgba_image, factor=0.5)
        validate_rgba(result_lower)
        assert result_lower.shape == rgba_image.shape

        # Test increasing saturation
        result_higher = adjust_saturation(rgba_image, factor=1.5)
        validate_rgba(result_higher)
        assert result_higher.shape == rgba_image.shape

        # Check that alpha channel is preserved
        assert np.allclose(result_lower[:, :, 3], rgba_image[:, :, 3], atol=1e-5)
        assert np.allclose(result_higher[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

        # Test zero saturation (should be grayscale with R=G=B)
        result_zero = adjust_saturation(rgba_image, factor=0.0)
        validate_rgba(result_zero)
        r_equals_g = np.allclose(result_zero[:, :, 0], result_zero[:, :, 1], atol=1e-5)
        g_equals_b = np.allclose(result_zero[:, :, 1], result_zero[:, :, 2], atol=1e-5)
        assert r_equals_g and g_equals_b

    def test_grayscale(self, rgba_image):
        """Test grayscale conversion."""
        result = grayscale(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Check if all RGB channels are equal (grayscale property)
        r_equals_g = np.allclose(result[:, :, 0], result[:, :, 1], atol=1e-5)
        g_equals_b = np.allclose(result[:, :, 1], result[:, :, 2], atol=1e-5)
        assert r_equals_g and g_equals_b

        # Check that alpha channel is preserved
        assert np.allclose(result[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

        # Should be the same as adjust_saturation with factor=0
        result2 = adjust_saturation(rgba_image, 0.0)
        assert np.allclose(result, result2, atol=1e-5)

    def test_invert_colors(self, rgba_image):
        """Test color inversion."""
        result = invert_colors(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Check if RGB channels are inverted but alpha is preserved
        expected = np.copy(rgba_image)
        expected[:, :, :3] = 1.0 - rgba_image[:, :, :3]
        assert np.allclose(result, expected, atol=1e-5)

        # Test on grayscale-converted image
        gray = grayscale(rgba_image)
        result_gray = invert_colors(gray)
        validate_rgba(result_gray)
        assert result_gray.shape == gray.shape

    def test_edge_detection(self, rgba_image, checkerboard_rgba):
        """Test edge detection."""
        # Test on RGBA image
        result = edge_detection(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Edge detection converts to grayscale internally but preserves alpha
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=1e-5)  # R = G
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=1e-5)  # G = B
        assert np.allclose(result[:, :, 3], rgba_image[:, :, 3], atol=1e-5)  # Alpha preserved

        # Edge detection should work particularly well on a checkerboard
        result_checker = edge_detection(checkerboard_rgba)
        validate_rgba(result_checker)
        assert result_checker.shape == checkerboard_rgba.shape

        # Checkerboard edges should have high values
        # Sum of edge detection result should be significant
        assert np.sum(result_checker[:, :, 0]) > 0.1 * result_checker.shape[0] * result_checker.shape[1]

        # Test with different strength
        # Instead of comparing maxed-out pixels, compare the overall edge intensity before clamping
        # Create a lower strength version for comparison
        result_weak = edge_detection(checkerboard_rgba, strength=0.25)
        result_strong = edge_detection(checkerboard_rgba, strength=1.0)

        # The stronger version should have a higher average edge value
        assert np.mean(result_strong[:, :, 0]) > np.mean(result_weak[:, :, 0])

    def test_vignette(self, rgba_image):
        """Test vignette effect."""
        # Test with default parameters
        result = vignette(rgba_image)
        validate_rgba(result)
        assert result.shape == rgba_image.shape

        # Check that alpha channel is preserved
        assert np.allclose(result[:, :, 3], rgba_image[:, :, 3], atol=1e-5)

        # Test with different strength and radius parameters
        result_custom = vignette(rgba_image, strength=0.9, radius=0.5)  # Much stronger vignette effect
        validate_rgba(result_custom)
        assert result_custom.shape == rgba_image.shape

        # Verify vignette effect by checking that edges are darker than center
        h, w = rgba_image.shape[:2]
        center_region = result[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        edge_region = result[0 : h // 6, 0 : w // 6]  # Top-left corner

        # Center should be brighter than edges (higher average RGB value)
        assert np.mean(center_region[:, :, :3]) > np.mean(edge_region[:, :, :3])

        # Get the same regions for the custom vignette
        center_custom = result_custom[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        edge_custom = result_custom[0 : h // 6, 0 : w // 6]

        # Calculate darkness difference between center and edge
        default_ratio = np.mean(edge_region[:, :, :3]) / np.mean(center_region[:, :, :3])
        custom_ratio = np.mean(edge_custom[:, :, :3]) / np.mean(center_custom[:, :, :3])

        # Custom vignette should have stronger effect (lower ratio between edge and center)
        assert custom_ratio < default_ratio
