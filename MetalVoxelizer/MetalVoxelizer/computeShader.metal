//
//  computeShader.metal
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

struct VoxelParams {
    float voxelSize;
    int gridSize;
};

kernel void generateGeometry(
    device Vertex *vertexBuffer [[ buffer(0) ]],
    device uint *indexBuffer [[ buffer(1) ]],
    device VoxelParams* params [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]) {

    float size = params->voxelSize;
    int gridSize = int(params->gridSize);
    float spacing = size * 1.1;

    int x = (id % gridSize) - ((gridSize-1)/2);
    int y = ((id / gridSize) % gridSize) - ((gridSize-1)/2);;
    int z = (id / (gridSize * gridSize)) - ((gridSize-1)/2);;

    float3 offset = float3(x, y, z);
    float3 basePos = offset*spacing;
    float4 color = float4(float(id % 3 == 0), float(id % 3 == 1), float(id % 3 == 2), 1.0);
    
    float3 v[8];
    v[0] = float3(-size * 0.5, -size * 0.5, size * 0.5);
    v[1] = float3(size * 0.5, -size * 0.5, size * 0.5);
    v[2] = float3(size * 0.5, -size * 0.5, -size * 0.5);
    v[3] = float3(-size * 0.5, -size * 0.5, -size * 0.5);
    v[4] = float3(-size * 0.5, size * 0.5, size * 0.5);
    v[5] = float3(size * 0.5, size * 0.5, size * 0.5);
    v[6] = float3(size * 0.5, size * 0.5, -size * 0.5);
    v[7] = float3(-size * 0.5, size * 0.5, -size * 0.5);
        
    Vertex verts[24];
    // bottom
    verts[0].position = basePos + v[0];
    verts[1].position = basePos + v[1];
    verts[2].position = basePos + v[2];
    verts[3].position = basePos + v[3];
    
    // Left
    verts[4].position = basePos + v[7];
    verts[5].position = basePos + v[4];
    verts[6].position = basePos + v[0];
    verts[7].position = basePos + v[3];
        
    // Front
    verts[8].position = basePos + v[4];
    verts[9].position = basePos + v[5];
    verts[10].position = basePos + v[1];
    verts[11].position = basePos + v[0];
        
    // Back
    verts[12].position = basePos + v[6];
    verts[13].position = basePos + v[7];
    verts[14].position = basePos + v[3];
    verts[15].position = basePos + v[2];
        
    // Right
    verts[16].position = basePos + v[5];
    verts[17].position = basePos + v[6];
    verts[18].position = basePos + v[2];
    verts[19].position = basePos + v[1];
        
    // Top
    verts[20].position = basePos + v[7];
    verts[21].position = basePos + v[6];
    verts[22].position = basePos + v[5];
    verts[23].position = basePos + v[4];

    float3 forward = float3(0.0,0.0,1.0);
    float3 back = float3(0.0,0.0,-1.0);
    float3 up = float3(0.0,1.0,0.0);
    float3 down = float3(0.0,-1.0,0.0);
    float3 right = float3(1.0,0.0,0.0);
    float3 left = float3(-1.0,0.0,0.0);
    
    verts[0].normal = down;
    verts[1].normal = down;
    verts[2].normal = down;
    verts[3].normal = down;
        
    verts[4].normal = left;
    verts[5].normal = left;
    verts[6].normal = left;
    verts[7].normal = left;
        
    verts[8].normal = forward;
    verts[9].normal = forward;
    verts[10].normal = forward;
    verts[11].normal = forward;
     
    verts[12].normal = back;
    verts[13].normal = back;
    verts[14].normal = back;
    verts[15].normal = back;
        
    verts[16].normal = right;
    verts[17].normal = right;
    verts[18].normal = right;
    verts[19].normal = right;
    
    verts[20].normal = up;
    verts[21].normal = up;
    verts[22].normal = up;
    verts[23].normal = up;
        
    for (uint i = 0; i < 24; ++i) {
        verts[i].color = color;
        vertexBuffer[id * 24 + i] = verts[i];
    }

    uint baseIndex = id * 36;
    uint vertexBase = id * 24;
    indexBuffer[baseIndex + 0] = vertexBase + 3;
    indexBuffer[baseIndex + 1] = vertexBase + 1;
    indexBuffer[baseIndex + 2] = vertexBase + 0;
    indexBuffer[baseIndex + 3] = vertexBase + 3;
    indexBuffer[baseIndex + 4] = vertexBase + 2;
    indexBuffer[baseIndex + 5] = vertexBase + 1;
    
    indexBuffer[baseIndex + 6] = vertexBase + 7;
    indexBuffer[baseIndex + 7] = vertexBase + 5;
    indexBuffer[baseIndex + 8] = vertexBase + 4;
    indexBuffer[baseIndex + 9] = vertexBase + 7;
    indexBuffer[baseIndex + 10] = vertexBase + 6;
    indexBuffer[baseIndex + 11] = vertexBase + 5;
        
    indexBuffer[baseIndex + 12] = vertexBase + 11;
    indexBuffer[baseIndex + 13] = vertexBase + 9;
    indexBuffer[baseIndex + 14] = vertexBase + 8;
    indexBuffer[baseIndex + 15] = vertexBase + 11;
    indexBuffer[baseIndex + 16] = vertexBase + 10;
    indexBuffer[baseIndex + 17] = vertexBase + 9;
        
    indexBuffer[baseIndex + 18] = vertexBase + 15;
    indexBuffer[baseIndex + 19] = vertexBase + 13;
    indexBuffer[baseIndex + 20] = vertexBase + 12;
    indexBuffer[baseIndex + 21] = vertexBase + 15;
    indexBuffer[baseIndex + 22] = vertexBase + 14;
    indexBuffer[baseIndex + 23] = vertexBase + 13;
        
    indexBuffer[baseIndex + 24] = vertexBase + 19;
    indexBuffer[baseIndex + 25] = vertexBase + 17;
    indexBuffer[baseIndex + 26] = vertexBase + 16;
    indexBuffer[baseIndex + 27] = vertexBase + 19;
    indexBuffer[baseIndex + 28] = vertexBase + 18;
    indexBuffer[baseIndex + 29] = vertexBase + 17;
        
    indexBuffer[baseIndex + 30] = vertexBase + 23;
    indexBuffer[baseIndex + 31] = vertexBase + 21;
    indexBuffer[baseIndex + 32] = vertexBase + 20;
    indexBuffer[baseIndex + 33] = vertexBase + 23;
    indexBuffer[baseIndex + 34] = vertexBase + 22;
    indexBuffer[baseIndex + 35] = vertexBase + 21;
}

/* // test quad rendering
kernel void generateGeometry(
    device Vertex *vertexBuffer [[ buffer(0) ]],
    device uint *indexBuffer [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]) {

    float size = 0.05;
    int gridSize = 10;
    float spacing = size * 2.0;

    int x = id % gridSize - ((gridSize-1)/2);
    int y = (id / gridSize) % gridSize;
    int z = id / (gridSize * gridSize);

    float3 offset = float3(x, y, z);
    float3 basePos = offset*spacing;
        basePos.x -= size;
    float4 color = float4(float(id % 3 == 0), float(id % 3 == 1), float(id % 3 == 2), 1.0);

    // 1 face (2 triangles), 4 vertices
    Vertex verts[4];
    verts[0].position = basePos + float3(-size*0.5, 0, 0);
    verts[1].position = basePos + float3(size*0.5, 0, 0);
    verts[2].position = basePos + float3(size*0.5, size, 0);
    verts[3].position = basePos + float3(-size*0.5, size, 0);

    for (uint i = 0; i < 4; ++i) {
        verts[i].normal = float3(0, 0, -1);
        verts[i].color = color;
        vertexBuffer[id * 4 + i] = verts[i];
    }

    // two triangles per quad
    uint baseIndex = id * 6;
    uint vertexBase = id * 4;
    indexBuffer[baseIndex + 0] = vertexBase + 0;
    indexBuffer[baseIndex + 1] = vertexBase + 1;
    indexBuffer[baseIndex + 2] = vertexBase + 2;
    indexBuffer[baseIndex + 3] = vertexBase + 2;
    indexBuffer[baseIndex + 4] = vertexBase + 3;
    indexBuffer[baseIndex + 5] = vertexBase + 0;
}
*/
