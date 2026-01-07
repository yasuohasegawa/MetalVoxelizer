#include <metal_stdlib>
using namespace metal;

struct Voxel {
    int3 position;
    uchar active;
    uchar3 padding;
    float4 color;
};

struct Vertex {
    float3 position;
    float3 normal;
    float4 color;
};

struct VoxelParams {
    float voxelSize;
    int gridSize;
};

kernel void compactActiveVoxels(
    device Voxel *voxelBuffer [[ buffer(0) ]],
    device uint *activeVoxelIndices [[ buffer(1) ]],
    device atomic_uint *activeCount [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]],
    uint totalThreads [[ threads_per_grid ]]) {
    
    if (id >= totalThreads) return;
    
    if (voxelBuffer[id].active == 1) {
        uint index = atomic_fetch_add_explicit(activeCount, 1, memory_order_relaxed);
        activeVoxelIndices[index] = id;
    }
}

inline bool isVoxelActive(device Voxel *voxelBuffer, int3 pos, int gridSize) {
    if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
        pos.x >= gridSize || pos.y >= gridSize || pos.z >= gridSize)
        return false;
    int index = pos.z * gridSize * gridSize + pos.y * gridSize + pos.x;
    return voxelBuffer[index].active == 1;
}

// Simplified geometry generation - no indirect command
kernel void generateGeometrySimple(
    device Vertex *vertexBuffer [[ buffer(0) ]],
    device uint *indexBuffer [[ buffer(1) ]],
    device Voxel *voxelBuffer [[ buffer(2) ]],
    device VoxelParams *params [[ buffer(3) ]],
    device uint *activeVoxelIndices [[ buffer(4) ]],
    device atomic_uint *activeCount [[ buffer(5) ]],
    uint threadId [[ thread_position_in_grid ]]) {
    
    uint totalActive = atomic_load_explicit(activeCount, memory_order_relaxed);
    
    if (threadId >= totalActive) return;
    
    uint voxelId = activeVoxelIndices[threadId];
    Voxel vox = voxelBuffer[voxelId];
    
    float size = params->voxelSize;
    int gridSize = int(params->gridSize);
    float spacing = size * 1.1;

    float3 offset = float3(vox.position.x - ((gridSize-1)/2),
                          vox.position.y - ((gridSize-1)/2),
                          vox.position.z - ((gridSize-1)/2));
    
    float3 basePos = offset * spacing;
    float4 color = vox.color;
    
    // Neighbor directions
    int3 directions[6] = {
        int3( 0, -1,  0), // bottom
        int3(-1,  0,  0), // left
        int3( 0,  0,  1), // front
        int3( 0,  0, -1), // back
        int3( 1,  0,  0), // right
        int3( 0,  1,  0)  // top
    };
    
    float3 normals[6] = {
        float3( 0, -1,  0),
        float3(-1,  0,  0),
        float3( 0,  0,  1),
        float3( 0,  0, -1),
        float3( 1,  0,  0),
        float3( 0,  1,  0)
    };
    
    // Cube corners
    float3 v[8];
    v[0] = float3(-size * 0.5, -size * 0.5,  size * 0.5);
    v[1] = float3( size * 0.5, -size * 0.5,  size * 0.5);
    v[2] = float3( size * 0.5, -size * 0.5, -size * 0.5);
    v[3] = float3(-size * 0.5, -size * 0.5, -size * 0.5);
    v[4] = float3(-size * 0.5,  size * 0.5,  size * 0.5);
    v[5] = float3( size * 0.5,  size * 0.5,  size * 0.5);
    v[6] = float3( size * 0.5,  size * 0.5, -size * 0.5);
    v[7] = float3(-size * 0.5,  size * 0.5, -size * 0.5);
    
    // Face definitions: which corners make each face
    int faceVerts[6][4] = {
        {3, 2, 1, 0}, // bottom
        {3, 0, 4, 7}, // left
        {0, 1, 5, 4}, // front
        {2, 3, 7, 6}, // back
        {1, 2, 6, 5}, // right
        {4, 5, 6, 7}  // top
    };
    
    uint vertexBase = threadId * 24;
    uint indexBase = threadId * 36;
    int3 p = int3(vox.position);
    
    // Generate each face
    for (int face = 0; face < 6; face++) {
        uint vOffset = vertexBase + face * 4;
        uint iOffset = indexBase + face * 6;
        
        // Check if face should be visible
        if (!isVoxelActive(voxelBuffer, p + directions[face], gridSize)) {
            // Add 4 vertices
            for (int i = 0; i < 4; i++) {
                int cornerIdx = faceVerts[face][i];
                vertexBuffer[vOffset + i].position = basePos + v[cornerIdx];
                vertexBuffer[vOffset + i].normal = normals[face];
                vertexBuffer[vOffset + i].color = color;
            }
            
            // Add 6 indices for 2 triangles
            indexBuffer[iOffset + 0] = vOffset + 0;
            indexBuffer[iOffset + 1] = vOffset + 1;
            indexBuffer[iOffset + 2] = vOffset + 2;
            indexBuffer[iOffset + 3] = vOffset + 0;
            indexBuffer[iOffset + 4] = vOffset + 2;
            indexBuffer[iOffset + 5] = vOffset + 3;
        } else {
            // Hidden face - write degenerate triangles
            for (int i = 0; i < 6; i++) {
                indexBuffer[iOffset + i] = vOffset;
            }
            
            for (int i = 0; i < 4; i++) {
                vertexBuffer[vOffset + i].position = float3(0, 0, 0);
                vertexBuffer[vOffset + i].normal = float3(0, 1, 0);
                vertexBuffer[vOffset + i].color = float4(0, 0, 0, 0);
            }
        }
    }
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
