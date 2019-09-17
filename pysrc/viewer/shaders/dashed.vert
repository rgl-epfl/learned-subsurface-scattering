#version 330

uniform mat4 modelViewProj;
in vec3 position;
in vec3 color;
in vec2 uv;
out vec4 frag_color;
out vec2 frag_uv;
out float invSegLength;

void main() {
    frag_color = vec4(color, 1.0);
    gl_Position = modelViewProj * vec4(position, 1.0);
    frag_uv = uv;
    int tmp = int(frag_uv.y * 100);
    tmp = 2 * (tmp / 2) + 1;
    invSegLength = float(tmp) / frag_uv.y;
}