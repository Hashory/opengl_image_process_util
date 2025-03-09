"""
Core functionality for OpenGL image processing.

This module provides the base functionality for setting up ModernGL contexts
and rendering images using OpenGL.
"""

import contextlib
from typing import Dict

import moderngl
import numpy as np

# Default vertex shader that maps texture coordinates from the quad to the fragment shader
DEFAULT_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

# Default fragment shader that simply samples from the input texture
DEFAULT_FRAGMENT_SHADER = """
#version 330

in vec2 v_texcoord;

uniform sampler2D texture0;

out vec4 f_color;

void main() {
    f_color = texture(texture0, v_texcoord);
}
"""

# Full screen quad vertices and texture coordinates
QUAD_VERTICES = np.array(
    [
        # Position (x, y), Texture coordinates (u, v)
        -1.0,
        -1.0,
        0.0,
        0.0,  # bottom-left
        1.0,
        -1.0,
        1.0,
        0.0,  # bottom-right
        -1.0,
        1.0,
        0.0,
        1.0,  # top-left
        1.0,
        1.0,
        1.0,
        1.0,  # top-right
    ],
    dtype=np.float32,
)


def validate_rgba(img: np.ndarray) -> None:
    """
    Validates that an image is in RGBA format (4 channels).

    Args:
        img: Input image as numpy ndarray, must be 3D with 4 channels

    Raises:
        ValueError: If the image is not in RGBA format
    """
    if img.ndim != 3:
        raise ValueError(f"Image must be 3D array with 4 channels, got {img.ndim}D array")

    if img.shape[2] != 4:
        raise ValueError(f"Image must have 4 channels (RGBA), got {img.shape[2]} channels")


class GLContext:
    """
    ModernGL context manager for image processing operations.

    Handles creation and management of the OpenGL context, programs, and resources.
    """

    def __init__(self, standalone: bool = True):
        """
        Initialize a new OpenGL context for image processing.

        Args:
            standalone: If True, creates a standalone context. If False, attempts to
                      use a shared context (useful within GUI applications).
        """
        self.ctx = moderngl.create_standalone_context() if standalone else moderngl.create_context()
        self.programs = {}
        self.vbos = {}
        self.fbos = {}
        self.textures = {}

        # Create a basic quad for rendering
        self.quad_vbo = self.ctx.buffer(QUAD_VERTICES.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.create_program(DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER),
            [(self.quad_vbo, "2f 2f", "in_position", "in_texcoord")],
        )

    def __del__(self):
        """Clean up OpenGL resources when the object is deleted."""
        self.release()

    def release(self):
        """Release all OpenGL resources."""
        for texture in self.textures.values():
            texture.release()
        for fbo in self.fbos.values():
            fbo.release()
        for vbo in self.vbos.values():
            vbo.release()
        for program in self.programs.values():
            program.release()

        if hasattr(self, "quad_vao"):
            self.quad_vao.release()
        if hasattr(self, "quad_vbo"):
            self.quad_vbo.release()

        self.textures = {}
        self.fbos = {}
        self.vbos = {}
        self.programs = {}

    def create_program(self, vertex_shader: str, fragment_shader: str) -> moderngl.Program:
        """
        Create a shader program from vertex and fragment shader source code.

        Args:
            vertex_shader: The vertex shader source code as a string
            fragment_shader: The fragment shader source code as a string

        Returns:
            A compiled and linked shader program
        """
        program_key = hash((vertex_shader, fragment_shader))

        if program_key not in self.programs:
            self.programs[program_key] = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        return self.programs[program_key]

    def create_texture(self, img: np.ndarray) -> moderngl.Texture:
        """
        Create an OpenGL texture from a numpy ndarray, requiring RGBA format.

        Args:
            img: Numpy array containing image data (HxWx4 format)

        Returns:
            An OpenGL texture object

        Raises:
            ValueError: If the image is not in RGBA format
        """
        # Validate image is RGBA format
        validate_rgba(img)

        # Image is guaranteed to be RGBA
        height, width, components = img.shape

        # Convert to float32 and ensure correct memory layout
        data = np.ascontiguousarray(img.astype(np.float32)).tobytes()

        # Create the texture
        texture = self.ctx.texture((width, height), components, data, dtype="f4")

        # Configure the texture
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False

        return texture

    def create_output_texture(self, width: int, height: int) -> moderngl.Texture:
        """
        Create an empty texture for rendering output.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels

        Returns:
            An empty OpenGL texture object with 4 components (RGBA)
        """
        components = 4  # Always RGBA

        texture = self.ctx.texture((width, height), components, dtype="f4")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False

        return texture

    def create_framebuffer(self, texture: moderngl.Texture) -> moderngl.Framebuffer:
        """
        Create a framebuffer with the given texture as a color attachment.

        Args:
            texture: An OpenGL texture to attach to the framebuffer

        Returns:
            A framebuffer object
        """
        return self.ctx.framebuffer(color_attachments=[texture])

    def process_image(
        self,
        img: np.ndarray,
        vertex_shader: str = DEFAULT_VERTEX_SHADER,
        fragment_shader: str = DEFAULT_FRAGMENT_SHADER,
        uniforms: dict = None,
        additional_textures: Dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process an image using the provided shaders, requiring RGBA in and out.

        Args:
            img: Input image as numpy ndarray (must be RGBA)
            vertex_shader: Vertex shader source code
            fragment_shader: Fragment shader source code
            uniforms: Dictionary of uniforms to pass to the shader program
            additional_textures: Dictionary of additional textures to bind, with uniform names as keys

        Returns:
            Processed image as numpy ndarray (RGBA)
        """
        # Validate input is RGBA
        validate_rgba(img)

        # Get dimensions
        height, width, components = img.shape

        # Fixed output components to 4 for RGBA
        out_components = 4

        # Create input texture
        input_texture = self.create_texture(img)

        # Create additional textures if provided, ensuring they're all RGBA
        tex_objects = {}
        if additional_textures:
            for name, tex_data in additional_textures.items():
                # Validate each additional texture
                validate_rgba(tex_data)
                tex_objects[name] = self.create_texture(tex_data)

        # Create output texture
        output_texture = self.create_output_texture(width, height)

        # Create framebuffer
        fbo = self.create_framebuffer(output_texture)

        # Create shader program
        program = self.create_program(vertex_shader, fragment_shader)

        # Set uniforms
        input_texture.use(0)
        program["texture0"] = 0

        # Bind additional textures
        tex_unit = 1
        for name, tex in tex_objects.items():
            if name in program:
                tex.use(tex_unit)
                program[name] = tex_unit
                tex_unit += 1

        if uniforms:
            for name, value in uniforms.items():
                if name in program:
                    program[name] = value

        # Bind framebuffer and render
        fbo.use()
        self.ctx.clear()

        # Create VAO with the specific program
        vao = self.ctx.vertex_array(program, [(self.quad_vbo, "2f 2f", "in_position", "in_texcoord")])
        vao.render(moderngl.TRIANGLE_STRIP)

        # Read the result
        result = np.frombuffer(fbo.read(components=out_components, dtype="f4"), dtype=np.float32)
        result = result.reshape(height, width, out_components)

        # Clean up
        vao.release()
        input_texture.release()
        for tex in tex_objects.values():
            tex.release()
        output_texture.release()
        fbo.release()

        return result


def render_to_ndarray(
    img: np.ndarray,
    vertex_shader: str = DEFAULT_VERTEX_SHADER,
    fragment_shader: str = DEFAULT_FRAGMENT_SHADER,
    uniforms: dict = None,
    additional_textures: Dict[str, np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience function to process an image with OpenGL shaders and return the result.

    Args:
        img: Input image as numpy ndarray (must be RGBA)
        vertex_shader: Vertex shader source code
        fragment_shader: Fragment shader source code
        uniforms: Dictionary of uniforms to pass to the shader program
        additional_textures: Dictionary of additional textures to bind, with uniform names as keys

    Returns:
        Processed image as numpy ndarray (RGBA)
    """
    with contextlib.ExitStack() as stack:
        # Create a context
        ctx = GLContext()
        stack.callback(ctx.release)

        # Process the image
        return ctx.process_image(
            img,
            vertex_shader,
            fragment_shader,
            uniforms,
            additional_textures,
        )
