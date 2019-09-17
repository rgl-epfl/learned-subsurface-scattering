#version 330
uniform mat4 modelViewProj;
uniform vec3 cameraPos;
in vec3 position;
in vec3 color;
out vec4 frag_color;
out vec3 uvw;
out vec3 viewVec;
out vec3 pos;

void main() {
    frag_color = vec4(color, 1.0);
    gl_Position = modelViewProj * vec4(position, 1.0);
    uvw = position.xyz;
    viewVec = normalize(-cameraPos + position.xyz);
    pos = position.xyz;
}