"""
Image processing effects using ModernGL.

This module provides various image processing effects implemented using OpenGL shaders.
"""

import numpy as np

from .core import DEFAULT_VERTEX_SHADER, render_to_ndarray, validate_rgba


def blur_image(img: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """
    Apply a Gaussian blur effect to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        radius: Blur radius, higher values create stronger blur

    Returns:
        Blurred image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    # Get image dimensions for shader uniforms
    height, width, _ = img.shape

    # Two-pass Gaussian blur for better performance
    # First horizontal pass - blur RGB only, preserve alpha
    horizontal_blur = """
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

    # Apply horizontal blur
    intermediate = render_to_ndarray(
        img,
        DEFAULT_VERTEX_SHADER,
        horizontal_blur,
        {"radius": radius, "image_size": (width, height)},
    )

    # Then vertical pass - blur RGB only, preserve alpha
    vertical_blur = """
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

    # Apply vertical blur to the intermediate result
    result = render_to_ndarray(
        intermediate,
        DEFAULT_VERTEX_SHADER,
        vertical_blur,
        {"radius": radius, "image_size": (width, height)},
    )

    # Make a copy to ensure the array is writable
    result = np.copy(result)

    # Ensure alpha channel is exactly preserved from the original image
    result[:, :, 3] = img[:, :, 3]

    return result


def sharpen_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply a sharpening filter to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Sharpening strength, higher values create stronger effect

    Returns:
        Sharpened image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    # Get image dimensions for shader uniforms
    height, width, _ = img.shape

    sharpen_shader = """
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

    result = render_to_ndarray(
        img,
        DEFAULT_VERTEX_SHADER,
        sharpen_shader,
        {"strength": strength, "image_size": (width, height)},
    )

    # Make a copy to ensure the array is writable (if needed for further processing)
    return np.copy(result)


def adjust_brightness(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust the brightness of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Brightness adjustment factor (0.0 = black, 1.0 = original, >1.0 = brighter)

    Returns:
        Brightness-adjusted image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    brightness_shader = """
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

    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, brightness_shader, {"factor": factor})
    # Make a copy to ensure the array is writable (if needed for further processing)
    return np.copy(result)


def adjust_contrast(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust the contrast of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Contrast adjustment factor (0.0 = gray, 1.0 = original, >1.0 = more contrast)

    Returns:
        Contrast-adjusted image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    contrast_shader = """
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

    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, contrast_shader, {"factor": factor})
    # Make a copy to ensure the array is writable (if needed for further processing)
    return np.copy(result)


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust the saturation of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        factor: Saturation adjustment factor (0.0 = grayscale, 1.0 = original, >1.0 = more saturated)

    Returns:
        Saturation-adjusted image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    saturation_shader = """
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

    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, saturation_shader, {"factor": factor})
    # Make a copy to ensure the array is writable (if needed for further processing)
    return np.copy(result)


def grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale, preserving alpha channel.

    Args:
        img: Input image as numpy ndarray (must be RGBA)

    Returns:
        Grayscale image as numpy ndarray (RGBA with R=G=B)
    """
    # Validate input is RGBA
    validate_rgba(img)

    # Use adjust_saturation with a factor of 0 for grayscale
    return adjust_saturation(img, 0.0)


def invert_colors(img: np.ndarray) -> np.ndarray:
    """
    Invert the colors of an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)

    Returns:
        Color-inverted image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    invert_shader = """
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

    result = render_to_ndarray(img, DEFAULT_VERTEX_SHADER, invert_shader)
    # Make a copy to ensure the array is writable (if needed for further processing)
    return np.copy(result)


def edge_detection(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply edge detection to an image using a Sobel filter.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Edge detection strength factor

    Returns:
        Edge-detected image as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    # First convert to grayscale (keeping RGBA format)
    gray_img = grayscale(img)

    # Get dimensions of the image
    height, width, _ = gray_img.shape

    edge_shader = """
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

    result = render_to_ndarray(
        gray_img,
        DEFAULT_VERTEX_SHADER,
        edge_shader,
        {"image_size": (width, height), "strength": strength},
    )

    # Make a copy to ensure the array is writable
    result = np.copy(result)

    # Ensure alpha channel is exactly preserved from the original image
    result[:, :, 3] = img[:, :, 3]

    return result


def vignette(img: np.ndarray, strength: float = 0.5, radius: float = 1.0) -> np.ndarray:
    """
    Apply a vignette effect to an image.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        strength: Vignette strength (0.0 to 1.0)
        radius: Vignette radius, distance from center to start darkening

    Returns:
        Image with vignette effect as numpy ndarray (RGBA)
    """
    # Validate input is RGBA
    validate_rgba(img)

    vignette_shader = """
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

    result = render_to_ndarray(
        img,
        DEFAULT_VERTEX_SHADER,
        vignette_shader,
        {"strength": strength, "radius": radius},
    )

    # Make a copy to ensure the array is writable
    result = np.copy(result)

    # Ensure alpha channel is exactly preserved from the original image
    result[:, :, 3] = img[:, :, 3]

    return result
