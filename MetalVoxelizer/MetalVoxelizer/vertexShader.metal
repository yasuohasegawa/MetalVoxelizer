//
//  vertexShader.metal
//  GPUVoxelRenderingTest
//
//  Created by Yasuo Hasegawa on 2025/06/05.
//

#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float3 position;
    float3 normal;
    float4 color;
};

struct Uniforms {
    float4x4 viewProjectionMatrix;
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float4 color;
};


vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                             device Vertex* vertexArray [[buffer(0)]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    VertexOut out;
    Vertex v = vertexArray[vertexID];
    out.position = uniforms.viewProjectionMatrix * float4(v.position, 1.0);
    //out.position.x /= uniforms.aspectRatio;
    out.color = v.color;
    out.normal = v.normal;
    return out;
}
