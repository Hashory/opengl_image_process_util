"""
Test fixtures for opengl_image_process_util tests.
"""

import os
import sys

import numpy as np
import pytest

# Add the parent directory to the system path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def rgba_image():
    """Create an RGBA test image with color gradients and transparency."""
    # Create a 128x128 RGBA image with red, green, blue gradients and alpha channel
    h, w = 128, 128
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Red channel: horizontal gradient
    red = xx
    # Green channel: vertical gradient
    green = yy
    # Blue channel: radial gradient from center
    cx, cy = 0.5, 0.5
    blue = 1 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * 1.4
    blue = np.clip(blue, 0, 1)

    # Alpha channel: circular gradient (1 in center, fading to 0.5 at edges)
    x_alpha = np.linspace(-1, 1, w)
    y_alpha = np.linspace(-1, 1, h)
    xx_alpha, yy_alpha = np.meshgrid(x_alpha, y_alpha)
    alpha = 1 - 0.5 * np.sqrt(xx_alpha**2 + yy_alpha**2)
    alpha = np.clip(alpha, 0.5, 1.0)  # Minimum 0.5 opacity

    # Combine channels into RGBA
    rgba = np.zeros((h, w, 4))
    rgba[:, :, 0] = red
    rgba[:, :, 1] = green
    rgba[:, :, 2] = blue
    rgba[:, :, 3] = alpha

    return rgba


@pytest.fixture
def checkerboard_rgba():
    """Create a checkerboard pattern test image in RGBA format."""
    h, w = 128, 128
    checkerboard = np.ones((h, w, 4))  # Initialize with all ones (including alpha=1)

    # Create 8x8 checkerboard pattern
    check_size = 16
    for i in range(h):
        for j in range(w):
            x = (i // check_size) % 2
            y = (j // check_size) % 2
            if (x + y) % 2 == 0:
                checkerboard[i, j, :3] = [1.0, 1.0, 1.0]  # White
            else:
                checkerboard[i, j, :3] = [0.0, 0.0, 0.0]  # Black

    # All pixels have full opacity
    return checkerboard


@pytest.fixture
def solid_color_rgba():
    """Create a solid color RGBA test image with varying alpha."""
    h, w = 128, 128
    solid = np.zeros((h, w, 4))

    # Set solid blue color
    solid[:, :, 2] = 0.8  # Blue at 80% intensity

    # Vary alpha horizontally from 0 to 1
    for j in range(w):
        solid[:, j, 3] = j / w

    return solid
