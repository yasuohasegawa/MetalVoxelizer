//
//  VoxelRenderer.swift
//  GPUVoxelRenderingTest
//
//  Created by Yasuo Hasegawa on 2025/06/05.
//

import SwiftUI
import MetalKit

struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var color: SIMD4<Float>
}

struct VoxelParams {
    var voxelSize:Float
    var gridSize:Int32
}

struct Uniforms {
    var viewProjectionMatrix: matrix_float4x4
}

class VoxelRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let computePipeline: MTLComputePipelineState
    let renderPipeline: MTLRenderPipelineState
    let depthStencilState: MTLDepthStencilState
    
    var vertexBuffer: MTLBuffer!
    var indexBuffer: MTLBuffer!
    var paramsBuffer: MTLBuffer!
    var indexCount: Int = 0
    var depthTexture: MTLTexture!
    
    var startTime: CFTimeInterval = CACurrentMediaTime()
    
    init?(mtkView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }

        self.device = device
        self.commandQueue = commandQueue
        mtkView.device = device

        // Load shaders
        let library = device.makeDefaultLibrary()!
        let computeFunc = library.makeFunction(name: "generateGeometry")!
        let vertexFunc = library.makeFunction(name: "vertex_main")!
        let fragmentFunc = library.makeFunction(name: "fragment_main")!

        computePipeline = try! device.makeComputePipelineState(function: computeFunc)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunc
        pipelineDescriptor.fragmentFunction = fragmentFunc
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        pipelineDescriptor.rasterSampleCount = 1
        
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3 // position
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float3 // normal
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[2].format = .float4 // color
        vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
        vertexDescriptor.attributes[2].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride

        pipelineDescriptor.vertexDescriptor = vertexDescriptor

        // deptrh setup
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
        depthStencilDescriptor.isDepthWriteEnabled = true
        
        depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
        
        renderPipeline = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)

        super.init()
        
        generateVoxelMesh()
        mtkView.delegate = self
    }

    func createDepthTexture(device: MTLDevice, width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.renderTarget]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }
    
    func generateVoxelMesh() {
        let gridSize = 100
        let voxelCount = gridSize * gridSize * gridSize
        let verticesPerQuad = 24
        let indicesPerQuad = 36

        let maxVertices = voxelCount * verticesPerQuad
        let maxIndices = voxelCount * indicesPerQuad

        var params = VoxelParams(
            voxelSize: 0.01,
            gridSize: Int32(gridSize)
        )
        
        vertexBuffer = device.makeBuffer(length: MemoryLayout<Vertex>.stride * maxVertices, options: .storageModePrivate)
        indexBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxIndices, options: .storageModePrivate)
        
        
        paramsBuffer = device.makeBuffer(bytes: &params,
                                             length: MemoryLayout<VoxelParams>.stride,
                                             options: [])
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(paramsBuffer, offset: 0, index: 2)

//        let threads = MTLSize(width: voxelCount, height: 1, depth: 1)
//        computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (voxelCount + 63) / 64, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        indexCount = maxIndices
    }

    func perspectiveFovRH(fovY: Float, aspect: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
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
    
    func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
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
    
    func makeRotationYMatrix(angle: Float) -> float4x4 {
        let c = cos(angle)
        let s = sin(angle)
        return float4x4([
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1,  0, 0),
            SIMD4<Float>(s, 0,  c, 0),
            SIMD4<Float>(0, 0,  0, 1)
        ])
    }
    
    func draw(in view: MTKView) {
        let descriptor = view.currentRenderPassDescriptor
        guard let descriptor = descriptor else { return }

        // Set or update depth texture
        if depthTexture == nil ||
            Int(view.drawableSize.width) != depthTexture!.width ||
            Int(view.drawableSize.height) != depthTexture!.height {
            depthTexture = createDepthTexture(
                device: device,
                width: Int(view.drawableSize.width),
                height: Int(view.drawableSize.height)
            )
        }

        descriptor.depthAttachment.texture = depthTexture
        descriptor.depthAttachment.loadAction = .clear
        descriptor.depthAttachment.storeAction = .store
        descriptor.depthAttachment.clearDepth = 1.0
        
        guard let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let fov: Float = .pi / 4
        let near: Float = 0.1
        let far: Float = 100.0

        let currentTime = CACurrentMediaTime()
        let elapsed = Float(currentTime - startTime)
        let speed = 0.5
        
        let modelMatrix = makeRotationYMatrix(angle: elapsed*Float(speed))
        let projection = perspectiveFovRH(fovY: fov, aspect: aspect, nearZ: near, farZ: far)
        let view = lookAt(
            eye: SIMD3<Float>(0, 0, 3.5),
            center: SIMD3<Float>(0, 0, 0),
            up: SIMD3<Float>(0, 1, 0)
        )
        
        let viewProj = projection * view * modelMatrix
        
        var uniforms = Uniforms(viewProjectionMatrix: viewProj)
        let uniformBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.stride, options: [])
        
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setVertexBuffer(paramsBuffer, offset: 0, index: 2)
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: indexCount,
                                      indexType: .uint32,
                                      indexBuffer: indexBuffer,
                                      indexBufferOffset: 0)
        
        encoder.setCullMode(.back)
        encoder.setFrontFacing(.counterClockwise)
        
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        //print(">>>>>> drawing")
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}


struct MetalView: UIViewRepresentable {
    class Coordinator {
        var renderer: VoxelRenderer?
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.clearColor = MTLClearColor(red: 0.7, green: 0.7, blue: 0.7, alpha: 1)
        mtkView.isOpaque = true
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = false
        mtkView.setNeedsDisplay()
        mtkView.preferredFramesPerSecond = 60
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float

        let renderer = VoxelRenderer(mtkView: mtkView)
        context.coordinator.renderer = renderer
        
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}




/*
 func generateQuadMesh() {
     let voxelCount = 10
     let verticesPerQuad = 4
     let indicesPerQuad = 6

     let maxVertices = voxelCount * verticesPerQuad
     let maxIndices = voxelCount * indicesPerQuad

     vertexBuffer = device.makeBuffer(length: MemoryLayout<Vertex>.stride * maxVertices, options: .storageModeShared)
     indexBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxIndices, options: .storageModeShared)
     
     guard let commandBuffer = commandQueue.makeCommandBuffer(),
           let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

     computeEncoder.setComputePipelineState(computePipeline)
     computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
     computeEncoder.setBuffer(indexBuffer, offset: 0, index: 1)

     let threads = MTLSize(width: voxelCount, height: 1, depth: 1)
     computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

     computeEncoder.endEncoding()
     commandBuffer.commit()
     commandBuffer.waitUntilCompleted()

     indexCount = maxIndices
 }
 */
