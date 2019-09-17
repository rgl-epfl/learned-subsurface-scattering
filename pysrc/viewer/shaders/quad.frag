#version 330

in vec2 uvFrag;
out vec4 color;

uniform sampler2D renderedTexture;

void main() {
    float far = 10000;
    float near = 0.001;
    float z = texture(renderedTexture, uvFrag).x;
    float linearDepth = (2.0 * near * far) / (far + near - z * (far - near));	
    color.xyz = vec3(linearDepth, linearDepth, linearDepth) / 100;
    color.w = 1.0;
}

