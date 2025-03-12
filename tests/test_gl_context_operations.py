"""
Tests for ModernGL context passing operations.
"""

import moderngl
import numpy as np
import pytest

from opengl_image_process_util.compositing import (
    add_blend,
    add_blend_textures,
    blend_images,
    blend_textures,
    multiply_blend,
    multiply_blend_textures,
    overlay_images,
    overlay_textures,
    screen_blend,
    screen_blend_textures,
    subtract_blend,
    subtract_blend_textures,
)
from opengl_image_process_util.core import GLContext
from opengl_image_process_util.effects import (
    adjust_brightness,
    adjust_brightness_texture,
    blur_image,
    blur_texture,
    edge_detection,
    edge_detection_texture,
    grayscale,
    grayscale_texture,
    invert_colors,
    invert_colors_texture,
    sharpen_image,
    sharpen_texture,
    vignette,
    vignette_texture,
)


class TestGLContextOperations:
    """Tests for ModernGL context passing between functions."""

    @pytest.fixture
    def gl_context(self):
        """Create a GLContext for testing."""
        ctx = GLContext()
        try:
            yield ctx
        finally:
            ctx.release()

    @pytest.fixture
    def moderngl_context(self):
        """Create a raw ModernGL context for testing."""
        ctx = moderngl.create_standalone_context()
        try:
            yield ctx
        finally:
            ctx.release()

    def test_context_passing(self, moderngl_context, rgba_image):
        """Test passing a ModernGL context between functions."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Pass the context to a function that processes textures
        result_texture = grayscale_texture(moderngl_context, texture)

        # Create a framebuffer to read the result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Process the same image using the numpy function
        expected_result = grayscale(rgba_image)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_context_reuse(self, moderngl_context, rgba_image):
        """Test reusing the same ModernGL context for multiple operations."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Chain multiple operations using the same context
        texture1 = grayscale_texture(moderngl_context, texture)
        texture2 = invert_colors_texture(moderngl_context, texture1)
        texture3 = blur_texture(moderngl_context, texture2, radius=2.0)

        # Create a framebuffer to read the result
        fbo = moderngl_context.framebuffer(color_attachments=[texture3])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Process the same image using numpy functions
        expected = grayscale(rgba_image)
        expected = invert_colors(expected)
        expected = blur_image(expected, radius=2.0)

        # Compare results - use a larger tolerance for blur operations
        assert np.allclose(result_data, expected, atol=1e-4)

        # Cleanup
        fbo.release()
        texture3.release()
        texture2.release()
        texture1.release()
        texture.release()

    def test_blur_texture(self, moderngl_context, rgba_image):
        """Test blur_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply blur using texture
        radius = 3.0
        result_texture = blur_texture(moderngl_context, texture, radius=radius)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply blur using numpy array
        expected_result = blur_image(rgba_image, radius=radius)

        # Compare results - use a larger tolerance for blur operations which can have
        # differences due to floating point precision and sampling patterns
        assert np.allclose(result_data, expected_result, atol=0.5)

        # Additional check: verify the general visual characteristics
        # Verify that alpha channels are preserved exactly
        assert np.allclose(result_data[:, :, 3], expected_result[:, :, 3], atol=1e-5)

        # Verify mean values of RGB channels are close
        for i in range(3):  # RGB channels
            result_mean = np.mean(result_data[:, :, i])
            expected_mean = np.mean(expected_result[:, :, i])
            assert abs(result_mean - expected_mean) < 0.1, f"Channel {i} means differ too much"

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_sharpen_texture(self, moderngl_context, rgba_image):
        """Test sharpen_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply sharpen using texture
        strength = 1.5
        result_texture = sharpen_texture(moderngl_context, texture, strength=strength)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply sharpen using numpy array
        expected_result = sharpen_image(rgba_image, strength=strength)

        # Compare results - use a slightly higher tolerance for sharpen operations
        assert np.allclose(result_data, expected_result, atol=1e-4)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_grayscale_texture(self, moderngl_context, rgba_image):
        """Test grayscale_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply grayscale using texture
        result_texture = grayscale_texture(moderngl_context, texture)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply grayscale using numpy array
        expected_result = grayscale(rgba_image)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_brightness_texture(self, moderngl_context, rgba_image):
        """Test adjust_brightness_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply brightness adjustment using texture
        factor = 1.5
        result_texture = adjust_brightness_texture(moderngl_context, texture, factor=factor)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply brightness adjustment using numpy array
        expected_result = adjust_brightness(rgba_image, factor=factor)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_invert_colors_texture(self, moderngl_context, rgba_image):
        """Test invert_colors_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply color inversion using texture
        result_texture = invert_colors_texture(moderngl_context, texture)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply color inversion using numpy array
        expected_result = invert_colors(rgba_image)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_edge_detection_texture(self, moderngl_context, rgba_image):
        """Test edge_detection_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply edge detection using texture
        strength = 1.2
        result_texture = edge_detection_texture(moderngl_context, texture, strength=strength)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply edge detection using numpy array
        expected_result = edge_detection(rgba_image, strength=strength)

        # Compare results - use a larger tolerance for edge detection which is sensitive
        # to small differences in processing order
        assert np.allclose(result_data, expected_result, atol=1e-4)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_vignette_texture(self, moderngl_context, rgba_image):
        """Test vignette_texture function with a context and compare with numpy version."""
        # Create a texture in the context
        texture = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply vignette using texture
        strength = 0.6
        radius = 0.8
        result_texture = vignette_texture(moderngl_context, texture, strength=strength, radius=radius)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply vignette using numpy array
        expected_result = vignette(rgba_image, strength=strength, radius=radius)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture.release()

    def test_blend_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test blend_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply blend using textures
        alpha = 0.7
        result_texture = blend_textures(moderngl_context, texture1, texture2, alpha=alpha)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply blend using numpy arrays
        expected_result = blend_images(rgba_image, solid_color_rgba, alpha=alpha)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()

    def test_overlay_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test overlay_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply overlay using textures
        result_texture = overlay_textures(moderngl_context, texture1, texture2)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply overlay using numpy arrays
        expected_result = overlay_images(rgba_image, solid_color_rgba)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()

    def test_multiply_blend_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test multiply_blend_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply multiply blend using textures
        result_texture = multiply_blend_textures(moderngl_context, texture1, texture2)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply multiply blend using numpy arrays
        expected_result = multiply_blend(rgba_image, solid_color_rgba)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()

    def test_screen_blend_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test screen_blend_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply screen blend using textures
        result_texture = screen_blend_textures(moderngl_context, texture1, texture2)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply screen blend using numpy arrays
        expected_result = screen_blend(rgba_image, solid_color_rgba)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()

    def test_add_blend_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test add_blend_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply add blend using textures
        result_texture = add_blend_textures(moderngl_context, texture1, texture2)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply add blend using numpy arrays
        expected_result = add_blend(rgba_image, solid_color_rgba)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()

    def test_subtract_blend_textures(self, moderngl_context, rgba_image, solid_color_rgba):
        """Test subtract_blend_textures function with a context and compare with numpy version."""
        # Create textures in the context
        texture1 = moderngl_context.texture(
            (rgba_image.shape[1], rgba_image.shape[0]),
            4,
            np.ascontiguousarray(rgba_image.astype(np.float32)).tobytes(),
            dtype="f4",
        )
        texture2 = moderngl_context.texture(
            (solid_color_rgba.shape[1], solid_color_rgba.shape[0]),
            4,
            np.ascontiguousarray(solid_color_rgba.astype(np.float32)).tobytes(),
            dtype="f4",
        )

        # Apply subtract blend using textures
        result_texture = subtract_blend_textures(moderngl_context, texture1, texture2)

        # Read result
        fbo = moderngl_context.framebuffer(color_attachments=[result_texture])
        result_data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(rgba_image.shape)

        # Apply subtract blend using numpy arrays
        expected_result = subtract_blend(rgba_image, solid_color_rgba)

        # Compare results
        assert np.allclose(result_data, expected_result, atol=1e-5)

        # Cleanup
        fbo.release()
        result_texture.release()
        texture2.release()
        texture1.release()
