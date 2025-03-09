"""
OpenGL Image Processing Utility

A library for GPU-accelerated image processing using ModernGL.
"""

from .compositing import (
    add_blend,
    alpha_composite,
    blend_images,
    multiply_blend,
    overlay_images,
    screen_blend,
    subtract_blend,
)
from .core import GLContext, render_to_ndarray
from .effects import (
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

__all__ = [
    "GLContext",
    "render_to_ndarray",
    "blur_image",
    "sharpen_image",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "grayscale",
    "invert_colors",
    "edge_detection",
    "vignette",
    "blend_images",
    "overlay_images",
    "alpha_composite",
    "multiply_blend",
    "screen_blend",
    "add_blend",
    "subtract_blend",
]
