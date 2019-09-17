#version 330

out vec4 color;
in vec4 frag_color;
in vec2 frag_uv;
in float invSegLength;

void main() {
    int tmp = int(frag_uv.x * invSegLength);
    color = frag_color * (1 - (tmp & 1));
}