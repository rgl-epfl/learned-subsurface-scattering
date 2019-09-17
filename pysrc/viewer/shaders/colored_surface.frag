#version 330

out vec4 color;
in vec3 frag_color;

void main() {    
    color = vec4(frag_color.r, frag_color.g, frag_color.b, 1.0);
}