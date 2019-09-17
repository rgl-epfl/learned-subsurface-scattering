#version 330

in vec2 uv;
in vec3 position;

out vec2 uvFrag;

void main() {
    gl_Position = vec4(position, 1.0);
    uvFrag = uv;
}
