#version 330

uniform mat4 modelViewProj;
in vec3 position;
in vec3 color;
out vec3 frag_color;

void main() {
    frag_color = color;
    gl_Position = modelViewProj * vec4(position, 1.0);
}