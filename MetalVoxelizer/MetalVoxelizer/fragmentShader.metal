//
//  fragmentShader.metal
//  GPUVoxelRenderingTest
//
//  Created by Yasuo Hasegawa on 2025/06/05.
//

#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float4 color;
};

fragment float4 fragment_main(VertexOut in [[ stage_in ]]) {
    float3 lightDir = normalize(float3(0.5, 0.5, 1)); // Directional light
    float3 normal = normalize(in.normal);              // Normalize normal
    
    float ambient = 0.2;
    float diff = max(dot(normal, lightDir), 0.0);      // Lambert diffuse term
    
    float3 baseColor = in.color.rgb;
    float3 finalColor = baseColor * (ambient + diff);  // Combine ambient + diffuse lighting
    
    return float4(finalColor, 1.0);
}
