"""
Image compositing functionality using ModernGL.

This module provides functions for blending and compositing images using OpenGL shaders.
"""

import numpy as np

from .core import DEFAULT_VERTEX_SHADER, render_to_ndarray, validate_rgba


def _validate_images(img1: np.ndarray, img2: np.ndarray):
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


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images together using a simple alpha blend.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)
        alpha: Blend factor (0.0 = only img1, 1.0 = only img2)

    Returns:
        Blended image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(img1, img2)

    blend_shader = """
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

    # Additional textures and uniforms for the shader
    additional_textures = {"texture1": img2}
    uniforms = {"alpha": alpha}

    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        blend_shader,
        uniforms,
        additional_textures,
    )


def overlay_images(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Overlay one image on top of another using alpha channel transparency.

    Args:
        base: Base image as numpy ndarray (must be RGBA)
        overlay: Overlay image as numpy ndarray (must be RGBA)

    Returns:
        Composited image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(base, overlay)

    overlay_shader = """
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

    # Additional textures for the shader
    additional_textures = {"texture1": overlay}

    return render_to_ndarray(
        base,
        DEFAULT_VERTEX_SHADER,
        overlay_shader,
        None,
        additional_textures,
    )


def alpha_composite(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
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


def multiply_blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Apply a multiply blend between two images.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Multiply-blended image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(img1, img2)

    multiply_shader = """
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

    # Additional textures for the shader
    additional_textures = {"texture1": img2}

    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        multiply_shader,
        None,
        additional_textures,
    )


def screen_blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Apply a screen blend between two images.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Screen-blended image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(img1, img2)

    screen_shader = """
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

    # Additional textures for the shader
    additional_textures = {"texture1": img2}

    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        screen_shader,
        None,
        additional_textures,
    )


def add_blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Add two images together, clamping to 1.0.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Added image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(img1, img2)

    add_shader = """
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

    # Additional textures for the shader
    additional_textures = {"texture1": img2}

    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        add_shader,
        None,
        additional_textures,
    )


def subtract_blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Subtract second image from first image, clamping to 0.0.

    Args:
        img1: First input image as numpy ndarray (must be RGBA)
        img2: Second input image as numpy ndarray (must be RGBA)

    Returns:
        Subtracted image as numpy ndarray (RGBA)
    """
    # Validate images
    _validate_images(img1, img2)

    subtract_shader = """
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

    # Additional textures for the shader
    additional_textures = {"texture1": img2}

    return render_to_ndarray(
        img1,
        DEFAULT_VERTEX_SHADER,
        subtract_shader,
        None,
        additional_textures,
    )
