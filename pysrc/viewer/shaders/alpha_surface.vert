#version 330

uniform mat4 modelViewProj;
uniform mat4 invTModelView;
in vec3 position;
in vec3 normal;
out vec3 frag_normal;
void main() {
    gl_Position = modelViewProj * vec4(position, 1.0);
    frag_normal = (invTModelView * vec4(normal, 0.0)).xyz;
}