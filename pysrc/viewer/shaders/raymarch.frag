#version 330

uniform sampler3D tex;
uniform sampler1D cmap;

uniform mat4 modelViewProj;

uniform vec3 cameraPos;
uniform vec3 bbMin;
uniform vec3 bbMax;
uniform vec3 voxelRes;

out vec4 color; 
in vec4 frag_color;
in vec3 uvw;
in vec3 viewVec;
in vec3 pos;

void main() {
    vec3 dir = normalize(-cameraPos + pos.xyz);

    vec4 col = vec4(0.0f);
    float maxDist = 5.0f;
    int nSteps = 1000;
    float stepSize = maxDist / nSteps;

    vec3 diag = bbMax - bbMin;
    diag = 1.0f / diag;

    float alpha = 1.0f; //0.4f;
    float accumAlpha = 0.0f;
    vec3 currentVoxel = vec3(-1);
    gl_FragDepth = 10000.0f;
    for (int i = 0; i < nSteps; ++i) {
        vec3 wsPos = pos + dir * i * stepSize;
        vec3 vsPos = diag * (wsPos - bbMin);
        vec3 voxelIndex = floor(vsPos * voxelRes);

        if (accumAlpha >= 1.0f) {     
            // Compute correct depth on last ray segment to correctly intersect with geometry
            vec4 vClipCoord = modelViewProj * vec4(wsPos, 1.0);
            float ndcDepth = vClipCoord.z / vClipCoord.w;
            gl_FragDepth = (1.0 - 0.0) * 0.5 * ndcDepth + (1.0 + 0.0) * 0.5;
            break;
        }

        vsPos = vsPos.zyx; // account for ordering in numpy being opposite of opengl
        vec3 val = texture(tex, vsPos).xyz;
        if (val.x > 0 || val.y > 0 || val.z > 0) {
            if (currentVoxel != voxelIndex) {
                color += vec4(val, 1.0) * alpha * (1 - accumAlpha);
                accumAlpha += (1 - accumAlpha) * alpha;
                currentVoxel = voxelIndex;
            }
        }
    }
    
    color.rgb = accumAlpha * texture(cmap, clamp(color.r, 0, 1 - 1.0f/64.0f)).rgb;      
    color.w = accumAlpha;
}