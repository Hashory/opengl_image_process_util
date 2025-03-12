"""
OpenGL Image Processing Utility

A library for GPU-accelerated image processing using ModernGL.
"""

from .compositing import (
    add_blend,
    add_blend_textures,
    alpha_composite,
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
from .core import GLContext, render_to_ndarray, render_to_texture
from .effects import (
    adjust_brightness,
    adjust_brightness_texture,
    adjust_contrast,
    adjust_contrast_texture,
    adjust_saturation,
    adjust_saturation_texture,
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

__all__ = [
    # Core functionality
    "GLContext",
    "render_to_ndarray",
    "render_to_texture",
    # Effects (ndarray-based)
    "blur_image",
    "sharpen_image",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "grayscale",
    "invert_colors",
    "edge_detection",
    "vignette",
    # Effects (texture-based)
    "blur_texture",
    "sharpen_texture",
    "adjust_brightness_texture",
    "adjust_contrast_texture",
    "adjust_saturation_texture",
    "grayscale_texture",
    "invert_colors_texture",
    "edge_detection_texture",
    "vignette_texture",
    # Compositing (ndarray-based)
    "blend_images",
    "overlay_images",
    "alpha_composite",
    "multiply_blend",
    "screen_blend",
    "add_blend",
    "subtract_blend",
    # Compositing (texture-based)
    "blend_textures",
    "overlay_textures",
    "multiply_blend_textures",
    "screen_blend_textures",
    "add_blend_textures",
    "subtract_blend_textures",
]
