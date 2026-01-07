#include <metal_stdlib>
using namespace metal;

// Line vertex structure
struct LineVertex {
    float3 position;
    float4 color;
};

struct Uniforms {
    float4x4 viewProjectionMatrix;
};

struct LineVertexOut {
    float4 position [[position]];
    float4 color;
};

// Line vertex shader
vertex LineVertexOut line_vertex_main(
    uint vertexID [[vertex_id]],
    constant LineVertex* vertexArray [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]]) {
    
    LineVertexOut out;
    LineVertex v = vertexArray[vertexID];
    out.position = uniforms.viewProjectionMatrix * float4(v.position, 1.0);
    out.color = v.color;
    return out;
}

// Line fragment shader
fragment float4 line_fragment_main(LineVertexOut in [[stage_in]]) {
    return in.color;
}
