"""
Image compositing functionality using ModernGL.
This module provides functions for blending and compositing images using OpenGL shaders.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import moderngl

from .core import (
    DEFAULT_VERTEX_SHADER,
    NDArray,
    ShaderSource,
    UniformDict,
    render_to_ndarray,
    render_to_texture,
    validate_rgba,
)

# Shader source code constants for different blend operations
BLEND_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float alpha;

out vec4 f_color;

void main() {
    vec4 color1 = texture(texture0, v_texcoord);
    vec4 color2 = texture(texture1, v_texcoord);
    
    // Simple linear interpolation
    f_color = mix(color1, color2, alpha);
}
"""

OVERLAY_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 f_color;

void main() {
    vec4 base_color = texture(texture0, v_texcoord);
    vec4 overlay_color = texture(texture1, v_texcoord);
    
    // Apply alpha blending
    float alpha = overlay_color.a;
    f_color.rgb = mix(base_color.rgb, overlay_color.rgb, alpha);
    f_color.a = max(base_color.a, overlay_color.a);
}
"""

MULTIPLY_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 f_color;

void main() {
    vec4 color1 = texture(texture0, v_texcoord);
    vec4 color2 = texture(texture1, v_texcoord);
    
    // Multiply blend
    vec3 blended = color1.rgb * color2.rgb;
    
    // Preserve alpha from base image
    f_color = vec4(blended, color1.a);
}
"""

SCREEN_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 f_color;

void main() {
    vec4 color1 = texture(texture0, v_texcoord);
    vec4 color2 = texture(texture1, v_texcoord);
    
    // Screen blend formula: 1 - (1 - a) * (1 - b)
    vec3 blended = 1.0 - (1.0 - color1.rgb) * (1.0 - color2.rgb);
    
    // Preserve alpha from base image
    f_color = vec4(blended, color1.a);
}
"""

ADD_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 f_color;

void main() {
    vec4 color1 = texture(texture0, v_texcoord);
    vec4 color2 = texture(texture1, v_texcoord);
    
    // Add colors and clamp
    vec3 blended = clamp(color1.rgb + color2.rgb, 0.0, 1.0);
    
    // Preserve alpha from base image
    f_color = vec4(blended, color1.a);
}
"""

SUBTRACT_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 f_color;

void main() {
    vec4 color1 = texture(texture0, v_texcoord);
    vec4 color2 = texture(texture1, v_texcoord);
    
    // Subtract colors and clamp
    vec3 blended = clamp(color1.rgb - color2.rgb, 0.0, 1.0);
    
    // Preserve alpha from base image
    f_color = vec4(blended, color1.a);
}
"""


def _validate_images(img1: NDArray, img2: NDArray) -> None:
    """
    Validate that two images can be composited together.
    Both images must be in RGBA format.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Raises:
        ValueError: If the images are not RGBA or have incompatible dimensions
    """
    # Validate both images are RGBA
    validate_rgba(img1)
    validate_rgba(img2)

    # Check for compatible shapes
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    if h1 != h2 or w1 != w2:
        raise ValueError(f"Images must have the same dimensions, got {(h1, w1)} and {(h2, w2)}")


def _apply_shader_to_images(
    img1: NDArray, img2: NDArray, shader: ShaderSource, uniforms: Optional[UniformDict] = None
) -> NDArray:
    """
    Apply a shader to two images with optional uniforms.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)
        shader: Fragment shader to apply
        uniforms: Optional shader uniforms

    Returns:
        Processed image as numpy ndarray (RGBA)
    """
    _validate_images(img1, img2)

    # Set up additional textures
    additional_textures = {"texture1": img2}

    # Process using the shader
    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        shader,
        uniforms,
        additional_textures,
    )


def _apply_shader_to_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
    shader: ShaderSource,
    uniforms: Optional[UniformDict] = None,
) -> "moderngl.Texture":
    """
    Apply a shader to two textures with optional uniforms.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture
        shader: Fragment shader to apply
        uniforms: Optional shader uniforms

    Returns:
        Processed ModernGL texture
    """
    # Set up additional textures
    additional_textures = {"texture1": texture2}

    # Process using the shader
    return render_to_texture(
        ctx,
        texture1,
        DEFAULT_VERTEX_SHADER,
        shader,
        uniforms,
        additional_textures,
    )


def blend_images(img1: NDArray, img2: NDArray, alpha: float = 0.5) -> NDArray:
    """
    Blend two images together using a simple alpha blend.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)
        alpha: Blend factor (0.0 = only img1, 1.0 = only img2)

    Returns:
        Blended image as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"alpha": alpha}

    # Apply blend shader
    return _apply_shader_to_images(img1, img2, BLEND_SHADER, uniforms)


def overlay_images(base: NDArray, overlay: NDArray) -> NDArray:
    """
    Overlay one image on top of another using alpha channel transparency.

    Args:
        base: Base image as numpy ndarray (must be RGBA)
        overlay: Overlay image as numpy ndarray (must be RGBA)

    Returns:
        Composited image as numpy ndarray (RGBA)
    """
    # Apply overlay shader
    return _apply_shader_to_images(base, overlay, OVERLAY_SHADER)


def alpha_composite(background: NDArray, foreground: NDArray) -> NDArray:
    """
    Compose a foreground image with alpha transparency over a background.

    Args:
        background: Background image as numpy ndarray (must be RGBA)
        foreground: Foreground image as numpy ndarray (must be RGBA)

    Returns:
        Composited image as numpy ndarray (RGBA)
    """
    # This is functionally the same as overlay_images
    return overlay_images(background, foreground)


def multiply_blend(img1: NDArray, img2: NDArray) -> NDArray:
    """
    Apply a multiply blend between two images.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Multiply-blended image as numpy ndarray (RGBA)
    """
    # Apply multiply shader
    return _apply_shader_to_images(img1, img2, MULTIPLY_SHADER)


def screen_blend(img1: NDArray, img2: NDArray) -> NDArray:
    """
    Apply a screen blend between two images.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Screen-blended image as numpy ndarray (RGBA)
    """
    # Apply screen shader
    return _apply_shader_to_images(img1, img2, SCREEN_SHADER)


def add_blend(img1: NDArray, img2: NDArray) -> NDArray:
    """
    Add two images together, clamping to 1.0.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Added image as numpy ndarray (RGBA)
    """
    # Apply add shader
    return _apply_shader_to_images(img1, img2, ADD_SHADER)


def subtract_blend(img1: NDArray, img2: NDArray) -> NDArray:
    """
    Subtract second image from first image, clamping to 0.0.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Subtracted image as numpy ndarray (RGBA)
    """
    # Apply subtract shader
    return _apply_shader_to_images(img1, img2, SUBTRACT_SHADER)


def blend_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
    alpha: float = 0.5,
) -> "moderngl.Texture":
    """
    Blend two textures together using a simple alpha blend.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture
        alpha: Blend factor (0.0 = only texture1, 1.0 = only texture2)

    Returns:
        Blended texture
    """
    # Set up uniforms
    uniforms = {"alpha": alpha}

    # Apply blend shader
    return _apply_shader_to_textures(ctx, texture1, texture2, BLEND_SHADER, uniforms)


def overlay_textures(
    ctx: "moderngl.Context",
    base: "moderngl.Texture",
    overlay: "moderngl.Texture",
) -> "moderngl.Texture":
    """
    Overlay one texture on top of another using alpha channel transparency.

    Args:
        ctx: ModernGL context
        base: Base texture
        overlay: Overlay texture

    Returns:
        Composited texture
    """
    # Apply overlay shader
    return _apply_shader_to_textures(ctx, base, overlay, OVERLAY_SHADER)


def multiply_blend_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
) -> "moderngl.Texture":
    """
    Apply a multiply blend between two textures.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture

    Returns:
        Multiply-blended texture
    """
    # Apply multiply shader
    return _apply_shader_to_textures(ctx, texture1, texture2, MULTIPLY_SHADER)


def screen_blend_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
) -> "moderngl.Texture":
    """
    Apply a screen blend between two textures.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture

    Returns:
        Screen-blended texture
    """
    # Apply screen shader
    return _apply_shader_to_textures(ctx, texture1, texture2, SCREEN_SHADER)


def add_blend_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
) -> "moderngl.Texture":
    """
    Add two textures together, clamping to 1.0.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture

    Returns:
        Added texture
    """
    # Apply add shader
    return _apply_shader_to_textures(ctx, texture1, texture2, ADD_SHADER)


def subtract_blend_textures(
    ctx: "moderngl.Context",
    texture1: "moderngl.Texture",
    texture2: "moderngl.Texture",
) -> "moderngl.Texture":
    """
    Subtract second texture from first texture, clamping to 0.0.

    Args:
        ctx: ModernGL context
        texture1: First input texture
        texture2: Second input texture

    Returns:
        Subtracted texture
    """
    # Apply subtract shader
    return _apply_shader_to_textures(ctx, texture1, texture2, SUBTRACT_SHADER)
