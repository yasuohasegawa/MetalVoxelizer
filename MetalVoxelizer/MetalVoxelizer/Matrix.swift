//
//  Matrix.swift
//  MetalVoxelizer
//
//  Created by Yasuo Hasegawa on 2025/06/06.
//
import MetalKit

class Matrix {
    static func perspectiveFovRH(fovY: Float, aspect: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
        let yScale = 1 / tan(fovY * 0.5)
        let xScale = yScale / aspect
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wzScale = -2 * farZ * nearZ / zRange

        return matrix_float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, zScale, -1),
            SIMD4<Float>(0, 0, wzScale, 0)
        ))
    }
    
    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
        let z = simd_normalize(eye - center)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)

        let t = SIMD3<Float>(
            -simd_dot(x, eye),
            -simd_dot(y, eye),
            -simd_dot(z, eye)
        )

        return matrix_float4x4(columns: (
            SIMD4<Float>(x.x, y.x, z.x, 0),
            SIMD4<Float>(x.y, y.y, z.y, 0),
            SIMD4<Float>(x.z, y.z, z.z, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        ))
    }
    
    static func makeRotationYMatrix(angle: Float) -> float4x4 {
        let c = cos(angle)
        let s = sin(angle)
        return float4x4([
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1,  0, 0),
            SIMD4<Float>(s, 0,  c, 0),
            SIMD4<Float>(0, 0,  0, 1)
        ])
    }
}
