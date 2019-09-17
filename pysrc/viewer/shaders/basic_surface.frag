#version 330

out vec4 color;
in vec3 frag_normal;

void main() {
    //color = vec4(frag_normal.x, frag_normal.y, frag_normal.z, 1); 
    color = vec4(frag_normal.z, frag_normal.z, frag_normal.z, 1); 
}