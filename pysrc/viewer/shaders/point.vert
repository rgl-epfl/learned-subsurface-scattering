#version 330

uniform mat4 modelViewProj;
in vec3 position;
in vec3 color;
out vec4 frag_color;
out vec3 pos;
out vec3 wsPos;

void main() {
    frag_color = vec4(color, 1.0);
    wsPos = position; // Note, this assumes we do not apply any model transform
    gl_Position = modelViewProj * vec4(position, 1.0);
    pos = gl_Position.xyz;
}