"""A minimal unlit shader in GLSL 120, because macOS GL supports only 120 or 150+ and ursina's built-in unlit shader (GLSL 130/140) fails to compile there and renders black."""
from ursina import Shader, Vec2

_VERT = """#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
attribute vec4 p3d_Color;
uniform vec2 texture_scale;
uniform vec2 texture_offset;
varying vec2 uvs;
varying vec4 vertex_color;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
}
"""

_FRAG = """#version 120
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
varying vec2 uvs;
varying vec4 vertex_color;
void main() {
    gl_FragColor = texture2D(p3d_Texture0, uvs) * p3d_ColorScale * vertex_color;
}
"""

mac_unlit_shader = Shader(
    name="mac_unlit_shader",
    language=Shader.GLSL,
    vertex=_VERT,
    fragment=_FRAG,
    default_input={"texture_scale": Vec2(1, 1), "texture_offset": Vec2(0, 0)},
)
