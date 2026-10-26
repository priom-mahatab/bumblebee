"""A minimal unlit shader written in GLSL 120.

Why this exists: ursina's built-in `unlit_shader` is GLSL 130 (vertex) / 140
(fragment), and **macOS supports neither** — Apple's GL stack offers only
GLSL 120 (legacy context) or 150+ (core profile), nothing in between. So on a
Mac ursina's shaders fail to compile ("version '130' is not supported") and
every surface renders black. GLSL 120 compiles on the stock macOS GL context,
so the sim shows its real colours/textures with no context changes.

It's a straight port of ursina's unlit shader (output = texture * colour-scale *
vertex colour), only down-levelled to 120 syntax (`attribute`/`varying`/
`texture2D`/`gl_FragColor`). Untextured entities get Panda's default white
sampler, so a plain coloured mesh shows its colour.
"""
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
