import simd
import Metal

struct Voxel {
    var position: SIMD3<Int32>
    var active: UInt8
    var padding: (UInt8, UInt8, UInt8) = (0, 0, 0)
    var color: SIMD4<Float>
}

struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var color: SIMD4<Float>
}

struct LineVertex {
    var position: SIMD3<Float>
    var color: SIMD4<Float>
}

struct VoxelParams {
    var voxelSize: Float
    var gridSize: Int32
}

struct Uniforms {
    var viewProjectionMatrix: matrix_float4x4
}

// Frustum plane for culling
struct Plane {
    var normal: SIMD3<Float>
    var distance: Float
    
    init(normal: SIMD3<Float>, distance: Float) {
        self.normal = normalize(normal)
        self.distance = distance
    }
    
    // Distance from point to plane (positive = in front)
    func distanceToPoint(_ point: SIMD3<Float>) -> Float {
        return simd_dot(normal, point) + distance
    }
}

// Axis-aligned bounding box
struct AABB {
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    
    var center: SIMD3<Float> {
        return (min + max) * 0.5
    }
    
    var extent: SIMD3<Float> {
        return (max - min) * 0.5
    }
    
    // Get all 8 corners of the bounding box
    func getCorners() -> [SIMD3<Float>] {
        return [
            SIMD3<Float>(min.x, min.y, min.z),
            SIMD3<Float>(max.x, min.y, min.z),
            SIMD3<Float>(min.x, max.y, min.z),
            SIMD3<Float>(max.x, max.y, min.z),
            SIMD3<Float>(min.x, min.y, max.z),
            SIMD3<Float>(max.x, min.y, max.z),
            SIMD3<Float>(min.x, max.y, max.z),
            SIMD3<Float>(max.x, max.y, max.z)
        ]
    }
}

// Frustum with 6 planes
struct Frustum {
    var planes: [Plane] = []
    
    init(viewProjectionMatrix: matrix_float4x4) {
        planes.reserveCapacity(6)
        
        // Extract frustum planes from view-projection matrix
        // Metal uses column-major matrices: matrix[column][row]
        let m = viewProjectionMatrix
        
        // Left plane: m3 + m0
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] + m[0][0],
                m[3][1] + m[0][1],
                m[3][2] + m[0][2]
            ),
            distance: m[3][3] + m[0][3]
        ))
        
        // Right plane: m3 - m0
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] - m[0][0],
                m[3][1] - m[0][1],
                m[3][2] - m[0][2]
            ),
            distance: m[3][3] - m[0][3]
        ))
        
        // Bottom plane: m3 + m1
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] + m[1][0],
                m[3][1] + m[1][1],
                m[3][2] + m[1][2]
            ),
            distance: m[3][3] + m[1][3]
        ))
        
        // Top plane: m3 - m1
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] - m[1][0],
                m[3][1] - m[1][1],
                m[3][2] - m[1][2]
            ),
            distance: m[3][3] - m[1][3]
        ))
        
        // Near plane: m3 + m2
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] + m[2][0],
                m[3][1] + m[2][1],
                m[3][2] + m[2][2]
            ),
            distance: m[3][3] + m[2][3]
        ))
        
        // Far plane: m3 - m2
        planes.append(Plane(
            normal: SIMD3<Float>(
                m[3][0] - m[2][0],
                m[3][1] - m[2][1],
                m[3][2] - m[2][2]
            ),
            distance: m[3][3] - m[2][3]
        ))
    }
    
    // Test if AABB is inside or intersecting frustum
    func intersects(aabb: AABB) -> Bool {
        // Add padding to make culling more conservative (prevents false culling)
        let padding: Float = 0.5  // Extra margin to prevent edge cases
        let expandedMin = aabb.min - SIMD3<Float>(repeating: padding)
        let expandedMax = aabb.max + SIMD3<Float>(repeating: padding)
        
        // Test against all 6 planes
        for plane in planes {
            // Find the positive vertex (corner furthest along plane normal)
            var positiveVertex = expandedMin
            if plane.normal.x >= 0 { positiveVertex.x = expandedMax.x }
            if plane.normal.y >= 0 { positiveVertex.y = expandedMax.y }
            if plane.normal.z >= 0 { positiveVertex.z = expandedMax.z }
            
            // If positive vertex is behind plane, AABB is completely outside
            if plane.distanceToPoint(positiveVertex) < 0 {
                return false
            }
        }
        
        // AABB is inside or intersecting frustum
        return true
    }
}
