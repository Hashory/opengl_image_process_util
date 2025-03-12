"""
Image processing effects using ModernGL.
This module provides various image processing effects implemented using OpenGL shaders.
"""

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import moderngl

from .core import (
    DEFAULT_VERTEX_SHADER,
    NDArray,
    ShaderSource,
    UniformDict,
    ensure_writable_copy,
    get_image_dimensions,
    preserve_alpha_channel,
    render_to_ndarray,
    render_to_texture,
    validate_rgba,
)

# Shader source code constants for different effects
# Horizontal Gaussian blur shader
HORIZONTAL_BLUR_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float radius;
uniform vec2 image_size;

out vec4 f_color;

void main() {
    vec2 pixel_size = 1.0 / image_size;
    vec3 sum_rgb = vec3(0.0);
    float weight_sum = 0.0;
    float original_alpha = texture(texture0, v_texcoord).a;
    
    // Calculate a reasonable number of samples based on radius
    int samples = int(min(radius * 3.0, 20.0));
    
    for (int i = -samples; i <= samples; i++) {
        float weight = exp(-float(i * i) / (2.0 * radius * radius));
        vec2 offset = vec2(float(i) * pixel_size.x, 0.0);
        sum_rgb += texture(texture0, v_texcoord + offset).rgb * weight;
        weight_sum += weight;
    }
    
    // Preserve the original alpha channel exactly
    f_color = vec4(sum_rgb / weight_sum, original_alpha);
}
"""

# Vertical Gaussian blur shader
VERTICAL_BLUR_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float radius;
uniform vec2 image_size;

out vec4 f_color;

void main() {
    vec2 pixel_size = 1.0 / image_size;
    vec3 sum_rgb = vec3(0.0);
    float weight_sum = 0.0;
    float original_alpha = texture(texture0, v_texcoord).a;
    
    // Calculate a reasonable number of samples based on radius
    int samples = int(min(radius * 3.0, 20.0));
    
    for (int i = -samples; i <= samples; i++) {
        float weight = exp(-float(i * i) / (2.0 * radius * radius));
        vec2 offset = vec2(0.0, float(i) * pixel_size.y);
        sum_rgb += texture(texture0, v_texcoord + offset).rgb * weight;
        weight_sum += weight;
    }
    
    // Preserve the original alpha channel exactly
    f_color = vec4(sum_rgb / weight_sum, original_alpha);
}
"""

# Sharpen shader
SHARPEN_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float strength;
uniform vec2 image_size;

out vec4 f_color;

void main() {
    vec2 pixel_size = 1.0 / image_size;
    
    // Sample center and neighbor pixels
    vec4 center = texture(texture0, v_texcoord);
    vec4 top = texture(texture0, v_texcoord + vec2(0.0, pixel_size.y));
    vec4 bottom = texture(texture0, v_texcoord - vec2(0.0, pixel_size.y));
    vec4 left = texture(texture0, v_texcoord - vec2(pixel_size.x, 0.0));
    vec4 right = texture(texture0, v_texcoord + vec2(pixel_size.x, 0.0));
    
    // Apply unsharp mask: 5*center - (top + bottom + left + right)
    vec4 sharpened = center * (1.0 + 4.0 * strength) - strength * (top + bottom + left + right);
    
    // Preserve the original alpha channel
    sharpened.a = center.a;
    
    // Clamp to valid range
    f_color = clamp(sharpened, 0.0, 1.0);
}
"""

# Brightness adjustment shader
BRIGHTNESS_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float factor;

out vec4 f_color;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    
    // Adjust RGB channels but keep alpha unchanged
    vec3 adjusted = color.rgb * factor;
    f_color = vec4(clamp(adjusted, 0.0, 1.0), color.a);
}
"""

# Contrast adjustment shader
CONTRAST_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float factor;

out vec4 f_color;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    
    // Calculate luminance for middle gray
    const vec3 luminance_weights = vec3(0.2126, 0.7152, 0.0722);
    float middle_gray = 0.5;
    
    // Apply contrast adjustment
    vec3 adjusted = mix(vec3(middle_gray), color.rgb, factor);
    f_color = vec4(clamp(adjusted, 0.0, 1.0), color.a);
}
"""

# Saturation adjustment shader
SATURATION_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float factor;

out vec4 f_color;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    
    // Convert to grayscale using luminance weights
    const vec3 luminance_weights = vec3(0.2126, 0.7152, 0.0722);
    float luminance = dot(color.rgb, luminance_weights);
    vec3 grayscale = vec3(luminance);
    
    // Mix between grayscale and original color based on saturation factor
    vec3 adjusted = mix(grayscale, color.rgb, factor);
    f_color = vec4(adjusted, color.a);
}
"""

# Invert colors shader
INVERT_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;

out vec4 f_color;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    
    // Invert RGB but keep alpha unchanged
    f_color = vec4(1.0 - color.rgb, color.a);
}
"""

# Edge detection shader
EDGE_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform vec2 image_size;
uniform float strength;

out vec4 f_color;

void main() {
    vec2 pixel_size = 1.0 / image_size;
    float original_alpha = texture(texture0, v_texcoord).a;
    
    // Sample neighboring pixels (using red channel since it's grayscale)
    float tl = texture(texture0, v_texcoord + pixel_size * vec2(-1.0, -1.0)).r;
    float t  = texture(texture0, v_texcoord + pixel_size * vec2( 0.0, -1.0)).r;
    float tr = texture(texture0, v_texcoord + pixel_size * vec2( 1.0, -1.0)).r;
    float l  = texture(texture0, v_texcoord + pixel_size * vec2(-1.0,  0.0)).r;
    float c  = texture(texture0, v_texcoord).r;
    float r  = texture(texture0, v_texcoord + pixel_size * vec2( 1.0,  0.0)).r;
    float bl = texture(texture0, v_texcoord + pixel_size * vec2(-1.0,  1.0)).r;
    float b  = texture(texture0, v_texcoord + pixel_size * vec2( 0.0,  1.0)).r;
    float br = texture(texture0, v_texcoord + pixel_size * vec2( 1.0,  1.0)).r;
    
    // Sobel operator
    float horiz = tl + 2.0 * l + bl - tr - 2.0 * r - br;
    float vert = tl + 2.0 * t + tr - bl - 2.0 * b - br;
    
    // Calculate edge magnitude and apply strength
    // The square root of the sum of squares gives us the gradient magnitude
    float edge = sqrt(horiz * horiz + vert * vert);
    
    // Apply the strength parameter (multiply before clamping)
    edge = edge * strength;
    
    // Clamp to valid range
    edge = clamp(edge, 0.0, 1.0);
    
    // Output edge detection result (grayscale with original alpha)
    f_color = vec4(edge, edge, edge, original_alpha);
}
"""

# Vignette shader
VIGNETTE_SHADER = """
#version 330

in vec2 v_texcoord;
uniform sampler2D texture0;
uniform float strength;
uniform float radius;

out vec4 f_color;

void main() {
    vec4 color = texture(texture0, v_texcoord);
    float original_alpha = color.a;
    
    // Calculate distance from center
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(v_texcoord, center);
    
    // Calculate vignette effect
    float vignette = smoothstep(radius, radius * 1.5, dist);
    vignette = 1.0 - (vignette * strength);
    
    // Apply vignette to RGB only
    f_color = vec4(color.rgb * vignette, original_alpha);
}
"""


def _apply_shader_with_dimensions(
    img: NDArray,
    shader: ShaderSource,
    uniforms: Optional[UniformDict] = None,
    additional_textures: Optional[Dict[str, NDArray]] = None,
    preserve_alpha: bool = False,
) -> NDArray:
    """
    Apply a shader to an image with dimensions automatically added to uniforms.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        shader: Fragment shader to apply
        uniforms: Optional shader uniforms
        additional_textures: Optional additional textures
        preserve_alpha: Whether to ensure alpha channel is preserved from original

    Returns:
        Processed image as numpy ndarray (RGBA)
    """
    # Validate input
    validate_rgba(img)

    # Get image dimensions
    height, width = get_image_dimensions(img)

    # Create a combined uniform dict with image dimensions
    combined_uniforms = uniforms.copy() if uniforms else {}

    # Add dimensions to uniforms if they're not already there
    if "image_size" not in combined_uniforms:
        combined_uniforms["image_size"] = (width, height)

    # Process the image with the shader
    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, shader, combined_uniforms, additional_textures)

    # Make a copy to ensure the array is writable
    result = ensure_writable_copy(result)

    # If requested, preserve alpha channel
    if preserve_alpha:
        preserve_alpha_channel(result, img)

    return result


def blur_image(img: NDArray, radius: float = 2.0) -> NDArray:
    """
    Apply a Gaussian blur effect to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        radius: Blur radius, higher values create stronger blur

    Returns:
        Blurred image as numpy ndarray (RGBA)
    """
    # Validate input
    validate_rgba(img)

    # First apply horizontal blur
    uniforms = {"radius": radius}
    intermediate = _apply_shader_with_dimensions(img, HORIZONTAL_BLUR_SHADER, uniforms)

    # Then apply vertical blur
    result = _apply_shader_with_dimensions(intermediate, VERTICAL_BLUR_SHADER, uniforms, preserve_alpha=True)

    return result


def sharpen_image(img: NDArray, strength: float = 1.0) -> NDArray:
    """
    Apply a sharpening filter to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Sharpening strength, higher values create stronger effect

    Returns:
        Sharpened image as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"strength": strength}

    # Apply sharpen shader
    return _apply_shader_with_dimensions(img, SHARPEN_SHADER, uniforms)


def adjust_brightness(img: NDArray, factor: float = 1.0) -> NDArray:
    """
    Adjust the brightness of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Brightness adjustment factor (0.0 = black, 1.0 = original, >1.0 = brighter)

    Returns:
        Brightness-adjusted image as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Validate input
    validate_rgba(img)

    # Apply brightness shader
    return render_to_ndarray(img, DEFAULT_VERTEX_SHADER, BRIGHTNESS_SHADER, uniforms)


def adjust_contrast(img: NDArray, factor: float = 1.0) -> NDArray:
    """
    Adjust the contrast of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Contrast adjustment factor (0.0 = gray, 1.0 = original, >1.0 = more contrast)

    Returns:
        Contrast-adjusted image as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Validate input
    validate_rgba(img)

    # Apply contrast shader
    return render_to_ndarray(img, DEFAULT_VERTEX_SHADER, CONTRAST_SHADER, uniforms)


def adjust_saturation(img: NDArray, factor: float = 1.0) -> NDArray:
    """
    Adjust the saturation of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Saturation adjustment factor (0.0 = grayscale, 1.0 = original, >1.0 = more saturated)

    Returns:
        Saturation-adjusted image as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Validate input
    validate_rgba(img)

    # Apply saturation shader
    return render_to_ndarray(img, DEFAULT_VERTEX_SHADER, SATURATION_SHADER, uniforms)


def grayscale(img: NDArray) -> NDArray:
    """
    Convert an image to grayscale, preserving alpha channel.

    Args:
        img: Input image as numpy ndarray (must be RGBA)

    Returns:
        Grayscale image as numpy ndarray (RGBA with R=G=B)
    """
    # Use adjust_saturation with a factor of 0 for grayscale
    return adjust_saturation(img, 0.0)


def invert_colors(img: NDArray) -> NDArray:
    """
    Invert the colors of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)

    Returns:
        Color-inverted image as numpy ndarray (RGBA)
    """
    # Validate input
    validate_rgba(img)

    # Apply invert shader
    return render_to_ndarray(img, DEFAULT_VERTEX_SHADER, INVERT_SHADER)


def edge_detection(img: NDArray, strength: float = 1.0) -> NDArray:
    """
    Apply edge detection to an image using a Sobel filter.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Edge detection strength factor

    Returns:
        Edge-detected image as numpy ndarray (RGBA)
    """
    # First convert to grayscale
    gray_img = grayscale(img)

    # Set up uniforms
    uniforms = {"strength": strength}

    # Apply edge detection
    result = _apply_shader_with_dimensions(gray_img, EDGE_SHADER, uniforms, preserve_alpha=True)

    return result


def vignette(img: NDArray, strength: float = 0.5, radius: float = 1.0) -> NDArray:
    """
    Apply a vignette effect to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Vignette strength (0.0 to 1.0)
        radius: Vignette radius, distance from center to start darkening

    Returns:
        Image with vignette effect as numpy ndarray (RGBA)
    """
    # Set up uniforms
    uniforms = {"strength": strength, "radius": radius}

    # Validate input
    validate_rgba(img)

    # Apply vignette shader
    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, VIGNETTE_SHADER, uniforms)

    # Make a copy to ensure the array is writable
    result = ensure_writable_copy(result)

    # Ensure alpha channel is exactly preserved
    preserve_alpha_channel(result, img)

    return result


def _apply_shader_to_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", shader: ShaderSource, uniforms: Optional[UniformDict] = None
) -> "moderngl.Texture":
    """
    Apply a shader to a texture with optional uniforms.

    Args:
        ctx: ModernGL context
        texture: Input texture
        shader: Fragment shader to apply
        uniforms: Optional shader uniforms

    Returns:
        Processed ModernGL texture
    """
    # Add texture dimensions to uniforms if they're not already there
    combined_uniforms = uniforms.copy() if uniforms else {}

    if "image_size" not in combined_uniforms:
        combined_uniforms["image_size"] = (texture.width, texture.height)

    # Process the texture with the shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, shader, combined_uniforms)


def blur_texture(ctx: "moderngl.Context", texture: "moderngl.Texture", radius: float = 2.0) -> "moderngl.Texture":
    """
    Apply a Gaussian blur effect to a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        radius: Blur radius, higher values create stronger blur

    Returns:
        Blurred texture
    """
    # Set up uniforms
    uniforms = {"radius": radius, "image_size": (texture.width, texture.height)}

    # First apply horizontal blur
    intermediate = _apply_shader_to_texture(ctx, texture, HORIZONTAL_BLUR_SHADER, uniforms)

    # Then apply vertical blur
    return _apply_shader_to_texture(ctx, intermediate, VERTICAL_BLUR_SHADER, uniforms)


def sharpen_texture(ctx: "moderngl.Context", texture: "moderngl.Texture", strength: float = 1.0) -> "moderngl.Texture":
    """
    Apply a sharpening filter to a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        strength: Sharpening strength, higher values create stronger effect

    Returns:
        Sharpened texture
    """
    # Set up uniforms
    uniforms = {"strength": strength, "image_size": (texture.width, texture.height)}

    # Apply sharpen shader
    return _apply_shader_to_texture(ctx, texture, SHARPEN_SHADER, uniforms)


def adjust_brightness_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", factor: float = 1.0
) -> "moderngl.Texture":
    """
    Adjust the brightness of a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        factor: Brightness adjustment factor (0.0 = black, 1.0 = original, >1.0 = brighter)

    Returns:
        Brightness-adjusted texture
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Apply brightness shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, BRIGHTNESS_SHADER, uniforms)


def adjust_contrast_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", factor: float = 1.0
) -> "moderngl.Texture":
    """
    Adjust the contrast of a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        factor: Contrast adjustment factor (0.0 = gray, 1.0 = original, >1.0 = more contrast)

    Returns:
        Contrast-adjusted texture
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Apply contrast shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, CONTRAST_SHADER, uniforms)


def adjust_saturation_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", factor: float = 1.0
) -> "moderngl.Texture":
    """
    Adjust the saturation of a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        factor: Saturation adjustment factor (0.0 = grayscale, 1.0 = original, >1.0 = more saturated)

    Returns:
        Saturation-adjusted texture
    """
    # Set up uniforms
    uniforms = {"factor": factor}

    # Apply saturation shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, SATURATION_SHADER, uniforms)


def grayscale_texture(ctx: "moderngl.Context", texture: "moderngl.Texture") -> "moderngl.Texture":
    """
    Convert a texture to grayscale.

    Args:
        ctx: ModernGL context
        texture: Input texture

    Returns:
        Grayscale texture
    """
    # Use adjust_saturation with a factor of 0 for grayscale
    return adjust_saturation_texture(ctx, texture, 0.0)


def invert_colors_texture(ctx: "moderngl.Context", texture: "moderngl.Texture") -> "moderngl.Texture":
    """
    Invert the colors of a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture

    Returns:
        Color-inverted texture
    """
    # Apply invert shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, INVERT_SHADER)


def edge_detection_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", strength: float = 1.0
) -> "moderngl.Texture":
    """
    Apply edge detection to a texture using a Sobel filter.

    Args:
        ctx: ModernGL context
        texture: Input texture
        strength: Edge detection strength factor

    Returns:
        Edge-detected texture
    """
    # First convert to grayscale
    gray_texture = grayscale_texture(ctx, texture)

    # Set up uniforms
    uniforms = {"strength": strength, "image_size": (texture.width, texture.height)}

    # Apply edge shader
    return _apply_shader_to_texture(ctx, gray_texture, EDGE_SHADER, uniforms)


def vignette_texture(
    ctx: "moderngl.Context", texture: "moderngl.Texture", strength: float = 0.5, radius: float = 1.0
) -> "moderngl.Texture":
    """
    Apply a vignette effect to a texture.

    Args:
        ctx: ModernGL context
        texture: Input texture
        strength: Vignette strength (0.0 to 1.0)
        radius: Vignette radius, distance from center to start darkening

    Returns:
        Texture with vignette effect
    """
    # Set up uniforms
    uniforms = {"strength": strength, "radius": radius}

    # Apply vignette shader
    return render_to_texture(ctx, texture, DEFAULT_VERTEX_SHADER, VIGNETTE_SHADER, uniforms)
