#version 330

uniform vec4 in_color;

out vec4 color;
in vec3 frag_normal;

void main() {
    //color = vec4(frag_normal.x, frag_normal.y, frag_normal.z, 1); 
    vec3 n = normalize(frag_normal);
    color = vec4(n.z * in_color.r, n.z * in_color.g, n.z * in_color.b, in_color.a); 
}