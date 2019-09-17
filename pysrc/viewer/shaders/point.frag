#version 330

uniform sampler2D renderedTexture;

uniform float far;
uniform float near;
uniform float sigmaT;
uniform vec2 screenSize;

uniform bool useRefPoint;
uniform bool cullOcclusions;
uniform vec3 refPoint;
uniform float refRadius;

out vec4 color;
in vec4 frag_color;
in vec3 pos;
in vec3 wsPos;

in vec4 gl_FragCoord;

void main() {
    color = vec4(0.0, 0.0, 0.0, 1.0);

    float z = texture(renderedTexture, gl_FragCoord.xy / screenSize).r;
    float linearDepth = (2.0 * near * far) / (far + near - z * (far - near));	
    float linearDepthCurrent = (2.0 * near * far) / (far + near - gl_FragCoord.z * (far - near));	
    linearDepthCurrent -= 0.01 * linearDepthCurrent;	

    if (linearDepthCurrent > linearDepth) {
        if (cullOcclusions)
            discard;
        color.rgb = 0.5 * frag_color.rgb + 0.5 * vec3(0.2, 0.2, 0.7);
    } else {
        color.rgb = frag_color.rgb;
        color.a = 1.0;
    }

    // Color the point based on its distance to a reference point
    if (useRefPoint && distance(wsPos, refPoint) < refRadius) {
        color.rgb = vec3(0.0, 1.0, 0.0);
    }

    vec2 coord = gl_PointCoord - vec2(0.5);
    float l = length(coord);
    vec3 normal = normalize(vec3(coord.x, coord.y, 1 - l));
    color.rgb =  color.rgb * normal.z;
    if (l > 0.5)
        discard;
}