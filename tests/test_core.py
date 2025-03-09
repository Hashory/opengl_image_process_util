"""
Tests for the core module of opengl_image_process_util.
"""

import numpy as np
import pytest

from opengl_image_process_util.core import (
    DEFAULT_VERTEX_SHADER,
    GLContext,
    render_to_ndarray,
    validate_rgba,
)


class TestValidateRgba:
    """Tests for the validate_rgba function."""

    def test_valid_rgba(self, rgba_image):
        """Test validation with valid RGBA image."""
        # Should not raise error
        validate_rgba(rgba_image)

    def test_invalid_dimensions(self):
        """Test validation with invalid dimension images."""
        # 2D array (grayscale)
        with pytest.raises(ValueError, match="must be 3D array"):
            validate_rgba(np.zeros((10, 10)))

        # 4D array
        with pytest.raises(ValueError, match="must be 3D array"):
            validate_rgba(np.zeros((10, 10, 3, 2)))

    def test_invalid_channels(self):
        """Test validation with invalid channel count."""
        # RGB (3 channels)
        with pytest.raises(ValueError, match="must have 4 channels"):
            validate_rgba(np.zeros((10, 10, 3)))

        # Single channel 3D array
        with pytest.raises(ValueError, match="must have 4 channels"):
            validate_rgba(np.zeros((10, 10, 1)))

        # Too many channels
        with pytest.raises(ValueError, match="must have 4 channels"):
            validate_rgba(np.zeros((10, 10, 5)))


class TestGLContext:
    """Tests for the GLContext class."""

    def test_init(self):
        """Test GLContext initialization."""
        ctx = GLContext()
        assert ctx.ctx is not None
        ctx.release()

    def test_create_program(self):
        """Test shader program creation."""
        ctx = GLContext()
        try:
            program = ctx.create_program(
                DEFAULT_VERTEX_SHADER,
                """
            #version 330
            
            in vec2 v_texcoord;
            uniform sampler2D texture0;
            
            out vec4 f_color;
            
            void main() {
                f_color = texture(texture0, v_texcoord);
            }
            """,
            )
            assert program is not None

            # Test caching - same program should be returned
            program2 = ctx.create_program(
                DEFAULT_VERTEX_SHADER,
                """
            #version 330
            
            in vec2 v_texcoord;
            uniform sampler2D texture0;
            
            out vec4 f_color;
            
            void main() {
                f_color = texture(texture0, v_texcoord);
            }
            """,
            )
            assert program is program2
        finally:
            ctx.release()

    def test_create_texture_rgba(self, rgba_image):
        """Test texture creation from RGBA image."""
        ctx = GLContext()
        try:
            texture = ctx.create_texture(rgba_image)
            assert texture is not None
            assert texture.width == rgba_image.shape[1]
            assert texture.height == rgba_image.shape[0]
            assert texture.components == 4  # RGBA has 4 components
        finally:
            ctx.release()

    def test_create_output_texture(self):
        """Test creation of output texture."""
        ctx = GLContext()
        try:
            width, height = 128, 128
            texture = ctx.create_output_texture(width, height)
            assert texture is not None
            assert texture.width == width
            assert texture.height == height
            assert texture.components == 4  # Should always be RGBA
        finally:
            ctx.release()


class TestRenderToNdarray:
    """Tests for the render_to_ndarray function."""

    def test_identity_transform(self, rgba_image):
        """Test that the default shaders don't change the image."""
        result = render_to_ndarray(rgba_image)
        assert result.shape == rgba_image.shape
        # Values should be very close (allow for some floating point error)
        assert np.allclose(result, rgba_image, atol=1e-5)

    def test_custom_shader(self, rgba_image):
        """Test rendering with a custom shader."""
        # Invert colors shader
        invert_shader = """
        #version 330
        
        in vec2 v_texcoord;
        uniform sampler2D texture0;
        
        out vec4 f_color;
        
        void main() {
            vec4 color = texture(texture0, v_texcoord);
            f_color = vec4(1.0 - color.rgb, color.a);  // Invert RGB, preserve alpha
        }
        """

        result = render_to_ndarray(rgba_image, DEFAULT_VERTEX_SHADER, invert_shader)
        assert result.shape == rgba_image.shape
        # Check if RGB colors are inverted (1 - original) but alpha is preserved
        expected_result = np.copy(rgba_image)
        expected_result[:, :, :3] = 1.0 - rgba_image[:, :, :3]
        assert np.allclose(result, expected_result, atol=1e-5)

    def test_uniforms(self, rgba_image):
        """Test passing uniforms to the shader."""
        # Brightness adjustment shader
        brightness_shader = """
        #version 330
        
        in vec2 v_texcoord;
        uniform sampler2D texture0;
        uniform float factor;
        
        out vec4 f_color;
        
        void main() {
            vec4 color = texture(texture0, v_texcoord);
            f_color = vec4(color.rgb * factor, color.a);  // Adjust RGB, preserve alpha
        }
        """

        factor = 0.5
        result = render_to_ndarray(rgba_image, DEFAULT_VERTEX_SHADER, brightness_shader, {"factor": factor})

        assert result.shape == rgba_image.shape
        # Check if brightness is adjusted for RGB but alpha is preserved
        expected_result = np.copy(rgba_image)
        expected_result[:, :, :3] = rgba_image[:, :, :3] * factor
        assert np.allclose(result, expected_result, atol=1e-5)

    def test_additional_textures(self, rgba_image):
        """Test rendering with additional textures."""
        # Create a second RGBA image (solid blue with 50% opacity)
        second_image = np.zeros_like(rgba_image)
        second_image[:, :, 2] = 1.0  # Blue
        second_image[:, :, 3] = 0.5  # 50% opacity

        # Blend shader
        blend_shader = """
        #version 330
        
        in vec2 v_texcoord;
        uniform sampler2D texture0;
        uniform sampler2D texture1;
        
        out vec4 f_color;
        
        void main() {
            vec4 color1 = texture(texture0, v_texcoord);
            vec4 color2 = texture(texture1, v_texcoord);
            
            // Simple alpha blending
            f_color = mix(color1, color2, color2.a);
        }
        """

        result = render_to_ndarray(
            rgba_image, DEFAULT_VERTEX_SHADER, blend_shader, additional_textures={"texture1": second_image}
        )

        assert result.shape == rgba_image.shape
        assert result.shape[2] == 4  # Should still be RGBA
