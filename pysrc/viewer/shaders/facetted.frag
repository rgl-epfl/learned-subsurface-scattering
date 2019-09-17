#version 330

out vec4 color;
in vec4 frag_color;

void main() {
    color = frag_color;
}