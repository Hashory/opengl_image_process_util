"""
Core functionality for OpenGL image processing.
This module provides the base functionality for setting up ModernGL contexts
and rendering images using OpenGL.
"""

import contextlib
from typing import Any, Dict, Optional, Tuple, TypeVar

import moderngl
import numpy as np

# Type aliases for cleaner type annotations
NDArray = np.ndarray
Texture = TypeVar("Texture", bound="moderngl.Texture")
GLContext = TypeVar("GLContext", bound="moderngl.Context")
ShaderSource = str
UniformDict = Dict[str, Any]
TextureDict = Dict[str, Any]  # Can be NDArray or moderngl.Texture depending on context

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


def validate_rgba(img: NDArray) -> None:
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


def get_image_dimensions(img: NDArray) -> Tuple[int, int]:
    """
    Extract the height and width from an image array.

    Args:
        img: Input image as numpy ndarray

    Returns:
        Tuple of (height, width)
    """
    height, width = img.shape[:2]
    return height, width


def ensure_writable_copy(result: NDArray) -> NDArray:
    """
    Ensure the array is writable by making a copy if necessary.

    Args:
        result: Input array

    Returns:
        Writable copy of the input array
    """
    return np.copy(result)


def preserve_alpha_channel(dest: NDArray, source: NDArray) -> None:
    """
    Copy alpha channel from source to destination array.

    Args:
        dest: Destination array to receive alpha channel
        source: Source array to provide alpha channel
    """
    dest[:, :, 3] = source[:, :, 3]


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
        self.quad_vao = self.create_vertex_array(
            self.create_program(DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER), self.quad_vbo
        )

    def __del__(self):
        """Clean up OpenGL resources when the object is deleted."""
        self.release()

    def release(self):
        """Release all OpenGL resources."""
        # Clean up resources in reverse order of dependency
        for texture in self.textures.values():
            texture.release()

        for fbo in self.fbos.values():
            fbo.release()

        for vbo in self.vbos.values():
            vbo.release()

        for program in self.programs.values():
            program.release()

        # Clean up the main quad resources
        if hasattr(self, "quad_vao"):
            self.quad_vao.release()

        if hasattr(self, "quad_vbo"):
            self.quad_vbo.release()

        # Clear the dictionaries
        self.textures = {}
        self.fbos = {}
        self.vbos = {}
        self.programs = {}

    def create_program(self, vertex_shader: ShaderSource, fragment_shader: ShaderSource) -> moderngl.Program:
        """
        Create a shader program from vertex and fragment shader source code.

        Args:
            vertex_shader: The vertex shader source code as a string
            fragment_shader: The fragment shader source code as a string

        Returns:
            A compiled and linked shader program
        """
        # Use hash as a cache key to avoid recompiling identical programs
        program_key = hash((vertex_shader, fragment_shader))

        if program_key not in self.programs:
            self.programs[program_key] = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        return self.programs[program_key]

    def create_vertex_array(self, program: moderngl.Program, vbo: moderngl.Buffer) -> moderngl.VertexArray:
        """
        Create a vertex array object with standard attributes for image rendering.

        Args:
            program: The shader program to use
            vbo: The vertex buffer object containing positions and texture coordinates

        Returns:
            A configured vertex array object
        """
        return self.ctx.vertex_array(program, [(vbo, "2f 2f", "in_position", "in_texcoord")])

    def create_texture(self, img: NDArray) -> moderngl.Texture:
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

        # Get image dimensions
        height, width = img.shape[:2]
        components = 4  # RGBA

        # Convert to float32 and ensure correct memory layout
        data = np.ascontiguousarray(img.astype(np.float32)).tobytes()

        # Create and configure the texture
        texture = self.ctx.texture((width, height), components, data, dtype="f4")
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

    def bind_textures_and_uniforms(
        self,
        program: moderngl.Program,
        input_texture: moderngl.Texture,
        additional_textures: Optional[Dict[str, moderngl.Texture]] = None,
        uniforms: Optional[UniformDict] = None,
    ) -> None:
        """
        Bind textures and set uniform variables in the shader program.

        Args:
            program: The shader program to bind to
            input_texture: The main input texture
            additional_textures: Optional dictionary of additional textures
            uniforms: Optional dictionary of uniform values
        """
        # Bind the main input texture
        input_texture.use(0)
        program["texture0"] = 0

        # Bind additional textures
        tex_unit = 1
        if additional_textures:
            for name, tex in additional_textures.items():
                if name in program:
                    tex.use(tex_unit)
                    program[name] = tex_unit
                    tex_unit += 1

        # Set uniforms
        if uniforms:
            for name, value in uniforms.items():
                if name in program:
                    program[name] = value

    def process_image(
        self,
        img: NDArray,
        vertex_shader: ShaderSource = DEFAULT_VERTEX_SHADER,
        fragment_shader: ShaderSource = DEFAULT_FRAGMENT_SHADER,
        uniforms: Optional[UniformDict] = None,
        additional_textures: Optional[Dict[str, NDArray]] = None,
    ) -> NDArray:
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
        height, width = img.shape[:2]
        out_components = 4  # Fixed output components to 4 for RGBA

        # Create input texture
        input_texture = self.create_texture(img)

        # Create additional textures if provided, ensuring they're all RGBA
        tex_objects = {}
        if additional_textures:
            for name, tex_data in additional_textures.items():
                validate_rgba(tex_data)
                tex_objects[name] = self.create_texture(tex_data)

        # Create output texture
        output_texture = self.create_output_texture(width, height)

        # Create framebuffer
        fbo = self.create_framebuffer(output_texture)

        # Create shader program
        program = self.create_program(vertex_shader, fragment_shader)

        # Bind textures and set uniforms
        self.bind_textures_and_uniforms(program, input_texture, tex_objects, uniforms)

        # Bind framebuffer and render
        fbo.use()
        self.ctx.clear()

        # Create VAO with the specific program
        vao = self.create_vertex_array(program, self.quad_vbo)
        vao.render(moderngl.TRIANGLE_STRIP)

        # Read the result
        result = np.frombuffer(fbo.read(components=out_components, dtype="f4"), dtype=np.float32).reshape(
            height, width, out_components
        )

        # Clean up
        vao.release()
        input_texture.release()
        for tex in tex_objects.values():
            tex.release()
        output_texture.release()
        fbo.release()

        return result

    def process_texture(
        self,
        input_texture: moderngl.Texture,
        vertex_shader: ShaderSource = DEFAULT_VERTEX_SHADER,
        fragment_shader: ShaderSource = DEFAULT_FRAGMENT_SHADER,
        uniforms: Optional[UniformDict] = None,
        additional_textures: Optional[Dict[str, moderngl.Texture]] = None,
    ) -> moderngl.Texture:
        """
        Process an OpenGL texture using the provided shaders.

        Args:
            input_texture: Input ModernGL texture
            vertex_shader: Vertex shader source code
            fragment_shader: Fragment shader source code
            uniforms: Dictionary of uniforms to pass to the shader program
            additional_textures: Dictionary of additional textures to bind, with uniform names as keys

        Returns:
            Processed ModernGL texture
        """
        # Create output texture
        output_texture = self.create_output_texture(input_texture.width, input_texture.height)

        # Create framebuffer
        fbo = self.create_framebuffer(output_texture)

        # Create shader program
        program = self.create_program(vertex_shader, fragment_shader)

        # Bind textures and set uniforms
        self.bind_textures_and_uniforms(program, input_texture, additional_textures, uniforms)

        # Bind framebuffer and render
        fbo.use()
        self.ctx.clear()

        # Create VAO with the specific program
        vao = self.create_vertex_array(program, self.quad_vbo)
        vao.render(moderngl.TRIANGLE_STRIP)

        # Clean up temporary resources
        vao.release()
        fbo.release()

        return output_texture


def render_to_ndarray(
    img: NDArray,
    vertex_shader: ShaderSource = DEFAULT_VERTEX_SHADER,
    fragment_shader: ShaderSource = DEFAULT_FRAGMENT_SHADER,
    uniforms: Optional[UniformDict] = None,
    additional_textures: Optional[Dict[str, NDArray]] = None,
) -> NDArray:
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
        # Create a context that will be properly cleaned up
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


def render_to_texture(
    ctx: moderngl.Context,
    input_texture: moderngl.Texture,
    vertex_shader: ShaderSource = DEFAULT_VERTEX_SHADER,
    fragment_shader: ShaderSource = DEFAULT_FRAGMENT_SHADER,
    uniforms: Optional[UniformDict] = None,
    additional_textures: Optional[Dict[str, moderngl.Texture]] = None,
) -> moderngl.Texture:
    """
    Process an OpenGL texture with shaders in an existing ModernGL context.

    Args:
        ctx: Existing ModernGL context
        input_texture: Input ModernGL texture
        vertex_shader: Vertex shader source code
        fragment_shader: Fragment shader source code
        uniforms: Dictionary of uniforms to pass to the shader program
        additional_textures: Dictionary of additional textures to bind, with uniform names as keys

    Returns:
        Processed ModernGL texture
    """
    # Create output texture in the same context
    output_texture = ctx.texture((input_texture.width, input_texture.height), 4, dtype="f4")
    output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    output_texture.repeat_x = False
    output_texture.repeat_y = False

    # Create framebuffer
    fbo = ctx.framebuffer(color_attachments=[output_texture])

    # Create program in this context
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    # Create quad buffer in the same context
    quad_vbo = ctx.buffer(QUAD_VERTICES.tobytes())

    try:
        # Set uniforms
        input_texture.use(0)
        program["texture0"] = 0

        # Bind additional textures
        tex_unit = 1
        if additional_textures:
            for name, tex in additional_textures.items():
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
        ctx.clear()

        # Create VAO with the specific program
        vao = ctx.vertex_array(program, [(quad_vbo, "2f 2f", "in_position", "in_texcoord")])
        vao.render(moderngl.TRIANGLE_STRIP)

        return output_texture
    finally:
        # Clean up resources in reverse order
        quad_vbo.release()
        program.release()
        fbo.release()
        # Don't release the output_texture as it's the return value
