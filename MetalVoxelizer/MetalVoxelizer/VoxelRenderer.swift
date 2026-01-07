import SwiftUI
import MetalKit

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

class VoxelRenderer: NSObject, MTKViewDelegate {
    let gridSize = 128
    let voxelSize: Float = 0.01
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let compactionPipeline: MTLComputePipelineState
    let geometryPipeline: MTLComputePipelineState
    let renderPipeline: MTLRenderPipelineState
    let linePipeline: MTLRenderPipelineState
    let depthStencilState: MTLDepthStencilState
    
    var vertexBuffer: MTLBuffer!
    var indexBuffer: MTLBuffer!
    var voxelBuffer: MTLBuffer!
    var paramsBuffer: MTLBuffer!
    var activeVoxelBuffer: MTLBuffer!
    var activeCountBuffer: MTLBuffer!
    var depthTexture: MTLTexture!
    
    // Grid and axis buffers
    var axisVertexBuffer: MTLBuffer!
    var axisIndexBuffer: MTLBuffer!
    var gridVertexBuffer: MTLBuffer!
    var gridIndexBuffer: MTLBuffer!
    var axisIndexCount: Int = 0
    var gridIndexCount: Int = 0
    
    var actualIndexCount: Int = 0
    
    private var lastPanLocation: CGPoint?
    private var cameraDistance: Float = 3.0
    private var cameraTarget = SIMD3<Float>(0, 0, 0)
    private var orientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0,1,0))
    
    init?(mtkView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }

        self.device = device
        self.commandQueue = commandQueue
        mtkView.device = device

        let library = device.makeDefaultLibrary()!
        let compactionFunc = library.makeFunction(name: "compactActiveVoxels")!
        let geometryFunc = library.makeFunction(name: "generateGeometrySimple")!
        let vertexFunc = library.makeFunction(name: "vertex_main")!
        let fragmentFunc = library.makeFunction(name: "fragment_main")!
        let lineVertexFunc = library.makeFunction(name: "line_vertex_main")!
        let lineFragmentFunc = library.makeFunction(name: "line_fragment_main")!
        
        compactionPipeline = try! device.makeComputePipelineState(function: compactionFunc)
        geometryPipeline = try! device.makeComputePipelineState(function: geometryFunc)

        // voxel render pipeline
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunc
        pipelineDescriptor.fragmentFunction = fragmentFunc
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float3
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[2].format = .float4
        vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
        vertexDescriptor.attributes[2].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride

        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Line render pipeline
        let linePipelineDescriptor = MTLRenderPipelineDescriptor()
        linePipelineDescriptor.vertexFunction = lineVertexFunc
        linePipelineDescriptor.fragmentFunction = lineFragmentFunc
        linePipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        linePipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let lineVertexDescriptor = MTLVertexDescriptor()
        lineVertexDescriptor.attributes[0].format = .float3  // position
        lineVertexDescriptor.attributes[0].offset = 0
        lineVertexDescriptor.attributes[0].bufferIndex = 0
        lineVertexDescriptor.attributes[1].format = .float4  // color
        lineVertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        lineVertexDescriptor.attributes[1].bufferIndex = 0
        lineVertexDescriptor.layouts[0].stride = MemoryLayout<LineVertex>.stride
        
        linePipelineDescriptor.vertexDescriptor = lineVertexDescriptor
        
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
        depthStencilDescriptor.isDepthWriteEnabled = true
        
        depthStencilState = device.makeDepthStencilState(descriptor: depthStencilDescriptor)!
        renderPipeline = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        linePipeline = try! device.makeRenderPipelineState(descriptor: linePipelineDescriptor)
        
        super.init()
        
        generateVoxelMesh()
        createThickAxisLines(thickness: 0.005)
        createGridLines()
        setupGestures(view: mtkView)
        mtkView.delegate = self
    }

    // Create RGB axis lines
    func createThickAxisLines(thickness: Float = 0.015) {
        var vertices: [Vertex] = []
        var indices: [UInt32] = []
        let axisSize: Float = 1.0
        
        // Helper: Create a rectangular prism from start to end
        func addAxisPrism(start: SIMD3<Float>, end: SIMD3<Float>, color: SIMD4<Float>) {
            let direction = end - start
            //let length = simd_length(direction)
            let normalizedDir = normalize(direction)
            
            // Determine which axis this is along
            let isXAxis = abs(normalizedDir.x) > 0.9
            let isYAxis = abs(normalizedDir.y) > 0.9
            //let isZAxis = abs(normalizedDir.z) > 0.9
            
            var corners: [SIMD3<Float>] = []
            
            if isXAxis {
                // X-axis: extend along X, make thick in Y and Z
                let t = thickness / 2
                corners = [
                    start + SIMD3<Float>(0, -t, -t),
                    start + SIMD3<Float>(0,  t, -t),
                    start + SIMD3<Float>(0,  t,  t),
                    start + SIMD3<Float>(0, -t,  t),
                    end   + SIMD3<Float>(0, -t, -t),
                    end   + SIMD3<Float>(0,  t, -t),
                    end   + SIMD3<Float>(0,  t,  t),
                    end   + SIMD3<Float>(0, -t,  t)
                ]
            } else if isYAxis {
                // Y-axis: extend along Y, make thick in X and Z
                let t = thickness / 2
                corners = [
                    start + SIMD3<Float>(-t, 0, -t),
                    start + SIMD3<Float>( t, 0, -t),
                    start + SIMD3<Float>( t, 0,  t),
                    start + SIMD3<Float>(-t, 0,  t),
                    end   + SIMD3<Float>(-t, 0, -t),
                    end   + SIMD3<Float>( t, 0, -t),
                    end   + SIMD3<Float>( t, 0,  t),
                    end   + SIMD3<Float>(-t, 0,  t)
                ]
            } else { // Z-axis
                let t = thickness / 2
                corners = [
                    start + SIMD3<Float>(-t, -t, 0),
                    start + SIMD3<Float>( t, -t, 0),
                    start + SIMD3<Float>( t,  t, 0),
                    start + SIMD3<Float>(-t,  t, 0),
                    end   + SIMD3<Float>(-t, -t, 0),
                    end   + SIMD3<Float>( t, -t, 0),
                    end   + SIMD3<Float>( t,  t, 0),
                    end   + SIMD3<Float>(-t,  t, 0)
                ]
            }
            
            let baseIdx = UInt32(vertices.count)
            
            // Add vertices with normals
            for corner in corners {
                vertices.append(Vertex(position: corner, normal: normalizedDir, color: color))
            }
            
            // Define the 6 faces of the box (12 triangles, 36 indices)
            let faceIndices: [UInt32] = [
                // Bottom face
                0, 1, 2,  0, 2, 3,
                // Top face
                4, 6, 5,  4, 7, 6,
                // Front face
                0, 4, 5,  0, 5, 1,
                // Back face
                2, 6, 7,  2, 7, 3,
                // Left face
                0, 3, 7,  0, 7, 4,
                // Right face
                1, 5, 6,  1, 6, 2
            ]
            
            for idx in faceIndices {
                indices.append(baseIdx + idx)
            }
        }
        
        // X-axis (Red)
        addAxisPrism(
            start: SIMD3<Float>(0, 0, 0),
            end: SIMD3<Float>(axisSize, 0, 0),
            color: SIMD4<Float>(1, 0, 0, 1)
        )
        
        // Y-axis (Green)
        addAxisPrism(
            start: SIMD3<Float>(0, 0, 0),
            end: SIMD3<Float>(0, axisSize, 0),
            color: SIMD4<Float>(0, 1, 0, 1)
        )
        
        // Z-axis (Blue)
        addAxisPrism(
            start: SIMD3<Float>(0, 0, 0),
            end: SIMD3<Float>(0, 0, axisSize),
            color: SIMD4<Float>(0, 0, 1, 1)
        )
        
        axisVertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<Vertex>.stride * vertices.count,
            options: [])
        
        axisIndexBuffer = device.makeBuffer(
            bytes: indices,
            length: MemoryLayout<UInt32>.stride * indices.count,
            options: [])
        
        axisIndexCount = indices.count
        
        print("Thick axis lines: \(vertices.count) vertices, \(indices.count) indices")
    }
    
    // Create grid lines
    func createGridLines() {
        let gridSizeCount = 64  // Number of grid lines
        let spacing: Float = 0.2
        
        var vertices: [LineVertex] = []
        var indices: [UInt16] = []
        
        let halfSize = Float(gridSizeCount) * spacing / 2
        var currentIndex: UInt16 = 0
        let lineColor = SIMD4<Float>(0.3, 0.3, 0.3, 0.3)
        
        for i in 0...gridSizeCount {
            let position = Float(i) * spacing - halfSize
            
            // Line parallel to X-axis (along Z)
            vertices.append(LineVertex(position: SIMD3<Float>(position, 0, -halfSize), color: lineColor))
            vertices.append(LineVertex(position: SIMD3<Float>(position, 0, halfSize), color: lineColor))
            indices.append(currentIndex)
            indices.append(currentIndex + 1)
            currentIndex += 2
            
            // Line parallel to Z-axis (along X)
            vertices.append(LineVertex(position: SIMD3<Float>(-halfSize, 0, position), color: lineColor))
            vertices.append(LineVertex(position: SIMD3<Float>(halfSize, 0, position), color: lineColor))
            indices.append(currentIndex)
            indices.append(currentIndex + 1)
            currentIndex += 2
        }
        
        gridVertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<LineVertex>.stride * vertices.count,
            options: [])
        
        gridIndexBuffer = device.makeBuffer(
            bytes: indices,
            length: MemoryLayout<UInt16>.stride * indices.count,
            options: [])
        
        gridIndexCount = indices.count
        
        print("Grid created: \(vertices.count) vertices, \(indices.count) indices")
    }
    
    // for vocel data to debug
    func createDummyVoxels(gridSize: Int) -> [Voxel] {
        var voxels = [Voxel]()
        voxels.reserveCapacity(gridSize * gridSize * gridSize)
        
        for z in 0..<gridSize {
            for y in 0..<gridSize {
                for x in 0..<gridSize {
                    let position = SIMD3<Int32>(Int32(x), Int32(y), Int32(z))
                    let random = Float.random(in: 0..<1)
                    let isActive: UInt8 = (random <= 0.1) ? 1 : 0
                    
                    let color: SIMD4<Float> = isActive == 1 ?
                        SIMD4<Float>(Float.random(in: 0.3..<1),
                                    Float.random(in: 0.3..<1),
                                    Float.random(in: 0.3..<1), 1) :
                        SIMD4<Float>(0.3, 0.3, 0.3, 0)
                    
                    let voxel = Voxel(position: position, active: isActive,
                                     padding: (0, 0, 0), color: color)
                    voxels.append(voxel)
                }
            }
        }
        return voxels
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
        let voxelCount = gridSize * gridSize * gridSize
        
        print("=== Mesh Generation Start ===")
        print("Total voxels: \(voxelCount)")
        
        var params = VoxelParams(voxelSize: voxelSize, gridSize: Int32(gridSize))
        
        // Create voxels first
        let voxels = createDummyVoxels(gridSize: gridSize)
        let activeCount = voxels.filter { $0.active == 1 }.count
        print("Active voxels: \(activeCount)")
        
        voxelBuffer = device.makeBuffer(
            bytes: voxels,
            length: MemoryLayout<Voxel>.stride * voxels.count,
            options: .storageModeShared)
        
        paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<VoxelParams>.stride,
            options: .storageModeShared)
        
        activeVoxelBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * voxelCount,
            options: .storageModeShared)
        
        var zero: UInt32 = 0
        activeCountBuffer = device.makeBuffer(
            bytes: &zero,
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared)
        
        // Step 1: Compact active voxels to find exact count
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        if let compactEncoder = commandBuffer.makeComputeCommandEncoder() {
            compactEncoder.setComputePipelineState(compactionPipeline)
            compactEncoder.setBuffer(voxelBuffer, offset: 0, index: 0)
            compactEncoder.setBuffer(activeVoxelBuffer, offset: 0, index: 1)
            compactEncoder.setBuffer(activeCountBuffer, offset: 0, index: 2)
            
            let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (voxelCount + 63) / 64, height: 1, depth: 1)
            compactEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            compactEncoder.endEncoding()
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read actual active count
        let countPtr = activeCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        let compactedCount = Int(countPtr.pointee)
        print("Compacted to \(compactedCount) active voxels")
        
        // NOW allocate buffers based on ACTUAL active count
        let actualMaxVertices = compactedCount * 24
        let actualMaxIndices = compactedCount * 36
        
        let vertexBufferSize = MemoryLayout<Vertex>.stride * actualMaxVertices
        let indexBufferSize = MemoryLayout<UInt32>.stride * actualMaxIndices
        
        print("Allocating vertex buffer: \(vertexBufferSize / 1024 / 1024) MB")
        print("Allocating index buffer: \(indexBufferSize / 1024 / 1024) MB")
        
        // Check if allocation would exceed limits
        let totalSize = vertexBufferSize + indexBufferSize
        if totalSize > 256 * 1024 * 1024 { // 256 MB safety limit
            print("WARNING: Buffer allocation would exceed safe limit!")
            print("Consider reducing grid size or voxel density")
        }
        
        // Allocate exact size needed for active voxels
        vertexBuffer = device.makeBuffer(
            length: vertexBufferSize,
            options: .storageModePrivate)
        
        indexBuffer = device.makeBuffer(
            length: indexBufferSize,
            options: .storageModePrivate)
        
        guard vertexBuffer != nil && indexBuffer != nil else {
            print("ERROR: Failed to allocate buffers!")
            print("Try reducing grid size or voxel density")
            return
        }
        
        print("Buffers allocated successfully")
        
        // Step 2: Generate geometry
        guard let commandBuffer2 = commandQueue.makeCommandBuffer() else { return }
        
        if let geometryEncoder = commandBuffer2.makeComputeCommandEncoder() {
            geometryEncoder.setComputePipelineState(geometryPipeline)
            geometryEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
            geometryEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
            geometryEncoder.setBuffer(voxelBuffer, offset: 0, index: 2)
            geometryEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)
            geometryEncoder.setBuffer(activeVoxelBuffer, offset: 0, index: 4)
            geometryEncoder.setBuffer(activeCountBuffer, offset: 0, index: 5)
            
            let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (compactedCount + 63) / 64, height: 1, depth: 1)
            geometryEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
            geometryEncoder.endEncoding()
        }
        
        commandBuffer2.commit()
        commandBuffer2.waitUntilCompleted()
        
        actualIndexCount = compactedCount * 36
        print("Generated \(actualIndexCount) indices")
        print("=== Mesh Generation Complete ===\n")
    }
    
    func draw(in view: MTKView) {
        guard actualIndexCount > 0 else { return }
        
        let descriptor = view.currentRenderPassDescriptor
        guard let descriptor = descriptor else { return }

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
        let near: Float = 0.01
        let far: Float = 100.0
        
        let forward = simd_act(orientation, SIMD3<Float>(0,0,-1))
        let up = simd_act(orientation, SIMD3<Float>(0,1,0))
        let right = simd_cross(forward, up)
        let eye = cameraTarget - forward * cameraDistance

        let r = SIMD4<Float>(right.x, up.x, -forward.x, 0)
        let u = SIMD4<Float>(right.y, up.y, -forward.y, 0)
        let f = SIMD4<Float>(right.z, up.z, -forward.z, 0)
        let p = SIMD4<Float>(-simd_dot(right, eye),
                             -simd_dot(up, eye),
                              simd_dot(forward, eye), 1)

        let viewMatrix = float4x4(columns:(r,u,f,p))
        let projection = Matrix.perspectiveFovRH(fovY: fov, aspect: aspect, nearZ: near, farZ: far)
        let modelMatrix = matrix_identity_float4x4
        let viewProj = projection * viewMatrix * modelMatrix
        
        var uniforms = Uniforms(viewProjectionMatrix: viewProj)
        let uniformBuffer = device.makeBuffer(bytes: &uniforms,
                                             length: MemoryLayout<Uniforms>.stride,
                                             options: [])
        
        // draw voxels
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setCullMode(.back)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: actualIndexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0)
        
        // draw grids
        encoder.setRenderPipelineState(linePipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setCullMode(.none)  // Lines don't need culling
        encoder.setVertexBuffer(gridVertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        
        encoder.drawIndexedPrimitives(
            type: .line,
            indexCount: gridIndexCount,
            indexType: .uint16,
            indexBuffer: gridIndexBuffer,
            indexBufferOffset: 0)
        
        // draw axis lines
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setCullMode(.back)
        encoder.setVertexBuffer(axisVertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: axisIndexCount,
            indexType: .uint32,
            indexBuffer: axisIndexBuffer,
            indexBufferOffset: 0)
        
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    private func setupGestures(view: MTKView) {
        let oneFingerPanGesture = UIPanGestureRecognizer(target: self, action: #selector(handleOneFingerPan(_:)))
        oneFingerPanGesture.maximumNumberOfTouches = 1
        view.addGestureRecognizer(oneFingerPanGesture)

        let twoFingerPanGesture = UIPanGestureRecognizer(target: self, action: #selector(handleTwoFingerPan(_:)))
        twoFingerPanGesture.minimumNumberOfTouches = 2
        view.addGestureRecognizer(twoFingerPanGesture)

        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        view.addGestureRecognizer(pinchGesture)
    }
    
    @objc private func handleOneFingerPan(_ gesture: UIPanGestureRecognizer) {
        if gesture.state == .began {
            lastPanLocation = gesture.location(in: gesture.view)
        } else if gesture.state == .changed {
            guard let last = lastPanLocation else { return }
            let currentLocation = gesture.location(in: gesture.view)

            let deltaX = Float(currentLocation.x - last.x) * 0.01
            let deltaY = Float(currentLocation.y - last.y) * 0.01
            let rotSpeed: Float = 0.5
            let yawQuat = simd_quatf(angle: -deltaX * rotSpeed, axis: SIMD3<Float>(0,1,0))
            let rightAxis = simd_act(orientation, SIMD3<Float>(1,0,0))
            let pitchQuat = simd_quatf(angle: -deltaY * rotSpeed, axis: rightAxis)

            orientation = yawQuat * pitchQuat * orientation
            lastPanLocation = currentLocation
        } else if gesture.state == .ended || gesture.state == .cancelled {
            lastPanLocation = nil
        }
    }
    
    @objc private func handleTwoFingerPan(_ gesture: UIPanGestureRecognizer) {
        if gesture.state == .changed {
            let translation = gesture.translation(in: gesture.view)
            let panSpeed: Float = 0.002 * cameraDistance
            let forward = simd_act(orientation, SIMD3<Float>(0,0,-1))
            let up = simd_act(orientation, SIMD3<Float>(0,1,0))
            let right = simd_cross(forward, up)

            cameraTarget -= right * Float(translation.x) * panSpeed
            cameraTarget += up * Float(translation.y) * panSpeed
            gesture.setTranslation(.zero, in: gesture.view)
        }
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        if gesture.state == .changed {
            cameraDistance /= Float(gesture.scale)
            cameraDistance = max(0.1, min(cameraDistance, 50))
            gesture.scale = 1.0
        }
    }
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
