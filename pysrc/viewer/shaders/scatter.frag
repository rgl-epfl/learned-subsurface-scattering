#version 330

uniform sampler3D tex;
uniform sampler1D cmap;

uniform mat4 modelViewProj;

uniform vec3 refPosition;
uniform vec3 bbMin;
uniform vec3 bbMax;
uniform vec3 voxelRes;

out vec4 color; 
in vec4 frag_color;
in vec3 uvw;
in vec3 viewVec;
in vec3 pos;


void main() {

    vec4 col = vec4(0.0f);
    float maxDist = 5.0f;

    vec3 diag = bbMax - bbMin;
    diag = 1.0f / diag;


    vec3 wsPos = pos;
    vec3 vsPos = diag * (wsPos - bbMin);

    // switch to numpy indexing
    vsPos = vsPos.zyx;
    
    vec3 d = refPosition - wsPos;
    float dist = sqrt(dot(d, d));
    // color.rgb = texture(cmap, clamp(exp(-dist), 0, 0.9)).rgb;
    if (dist < 0.01)
        color.rgb = vec3(1, 0, 0);
    else
        color.rgb = texture(cmap, clamp(texture(tex, vsPos).r, 0, 0.9)).rgb;
    color.a = 1.0;
}