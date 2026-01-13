import SwiftUI
import MetalKit

// Ray structure for ray-casting
struct Ray {
    var origin: SIMD3<Float>
    var direction: SIMD3<Float>
    
    func pointAt(distance: Float) -> SIMD3<Float> {
        return origin + direction * distance
    }
}

// Ray-voxel intersection result
struct RayVoxelIntersection {
    var hit: Bool
    var voxelPosition: SIMD3<Int32>
    var distance: Float
    var normal: SIMD3<Float>  // Face normal that was hit
}

class VoxelRenderer: NSObject, MTKViewDelegate {
    let gridSize = 128
    let voxelSize: Float = 0.01
    
    // Buffer splitting configuration
    let maxBufferSize = 200 * 1024 * 1024
    let maxVoxelsPerChunk: Int
    let enableFrustumCulling = true
    let debugFrustumCulling = false
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let compactionPipeline: MTLComputePipelineState
    let geometryPipeline: MTLComputePipelineState
    let renderPipeline: MTLRenderPipelineState
    let linePipeline: MTLRenderPipelineState
    let depthStencilState: MTLDepthStencilState
    
    var vertexBuffers: [MTLBuffer] = []
    var indexBuffers: [MTLBuffer] = []
    var indexCounts: [Int] = []
    var chunkOffsets: [Int] = []
    var chunkBounds: [AABB] = []
    
    var voxelBuffer: MTLBuffer!
    var paramsBuffer: MTLBuffer!
    var activeVoxelBuffer: MTLBuffer!
    var activeCountBuffer: MTLBuffer!
    var depthTexture: MTLTexture!
    
    var axisVertexBuffer: MTLBuffer!
    var axisIndexBuffer: MTLBuffer!
    var gridVertexBuffer: MTLBuffer!
    var gridIndexBuffer: MTLBuffer!
    var axisIndexCount: Int = 0
    var gridIndexCount: Int = 0
    
    var totalIndexCount: Int = 0
    
    private var lastPanLocation: CGPoint?
    private var cameraDistance: Float = 3.0
    private var cameraTarget = SIMD3<Float>(0, 0, 0)
    private var orientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0,1,0))
    
    // Edit controller reference
    weak var editController: VoxelEditController?
    private var needsRebuild = false
    
    // Store voxel data in memory for editing
    private var voxelData: [Voxel] = []
    
    init?(mtkView: MTKView, editController: VoxelEditController?) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }

        self.device = device
        self.commandQueue = commandQueue
        self.editController = editController
        
        let vertexSize = MemoryLayout<Vertex>.stride * 24
        let indexSize = MemoryLayout<UInt32>.stride * 36
        let totalSizePerVoxel = vertexSize + indexSize
        self.maxVoxelsPerChunk = maxBufferSize / totalSizePerVoxel
        
        print("Max voxels per chunk: \(maxVoxelsPerChunk)")
        
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
        
        let linePipelineDescriptor = MTLRenderPipelineDescriptor()
        linePipelineDescriptor.vertexFunction = lineVertexFunc
        linePipelineDescriptor.fragmentFunction = lineFragmentFunc
        linePipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        linePipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let lineVertexDescriptor = MTLVertexDescriptor()
        lineVertexDescriptor.attributes[0].format = .float3
        lineVertexDescriptor.attributes[0].offset = 0
        lineVertexDescriptor.attributes[0].bufferIndex = 0
        lineVertexDescriptor.attributes[1].format = .float4
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

    // MARK: - Ray Casting
    
    func createRayFromScreenPoint(_ point: CGPoint, viewSize: CGSize) -> Ray {
        let aspect = Float(viewSize.width / viewSize.height)
        let fov: Float = .pi / 4
        
        let forward = simd_act(orientation, SIMD3<Float>(0,0,-1))
        let up = simd_act(orientation, SIMD3<Float>(0,1,0))
        let right = simd_cross(forward, up)
        let eye = cameraTarget - forward * cameraDistance
        
        // Convert screen coordinates to NDC (-1 to 1)
        let x = (Float(point.x) / Float(viewSize.width)) * 2.0 - 1.0
        let y = 1.0 - (Float(point.y) / Float(viewSize.height)) * 2.0
        
        // Calculate ray direction
        let tanHalfFov = tan(fov / 2.0)
        let rayDir = normalize(
            right * (x * aspect * tanHalfFov) +
            up * (y * tanHalfFov) +
            forward
        )
        
        return Ray(origin: eye, direction: rayDir)
    }
    
    func raycastVoxels(_ ray: Ray) -> RayVoxelIntersection {
        let spacing = voxelSize * 1.1
        let halfGrid = Float(gridSize - 1) / 2.0
        
        var closestHit = RayVoxelIntersection(
            hit: false,
            voxelPosition: SIMD3<Int32>(0, 0, 0),
            distance: Float.infinity,
            normal: SIMD3<Float>(0, 0, 0)
        )
        
        // DDA algorithm - step through voxel grid
        let maxDistance: Float = 20.0
        let stepSize: Float = voxelSize * 0.3  // Smaller steps for accuracy
        var currentDistance: Float = 0.0
        
        while currentDistance < maxDistance {
            let point = ray.pointAt(distance: currentDistance)
            
            // Convert world position to voxel grid position
            // USE FLOOR, NOT ROUND - we want the voxel we're currently in
            let voxelX = Int32(floor(point.x / spacing + halfGrid))
            let voxelY = Int32(floor(point.y / spacing + halfGrid))
            let voxelZ = Int32(floor(point.z / spacing + halfGrid))
            
            // Check if within grid bounds
            if voxelX >= 0 && voxelX < gridSize &&
               voxelY >= 0 && voxelY < gridSize &&
               voxelZ >= 0 && voxelZ < gridSize {
                
                let voxelIndex = Int(voxelZ * Int32(gridSize * gridSize) + voxelY * Int32(gridSize) + voxelX)
                
                if voxelIndex < voxelData.count && voxelData[voxelIndex].active == 1 {
                    // Hit! Now calculate which face we hit
                    let voxelWorldPos = SIMD3<Float>(
                        (Float(voxelX) - halfGrid) * spacing,
                        (Float(voxelY) - halfGrid) * spacing,
                        (Float(voxelZ) - halfGrid) * spacing
                    )
                    
                    // Calculate relative position within voxel (-0.5 to 0.5)
                    let relativePos = (point - voxelWorldPos) / voxelSize
                    
                    // Find which face is closest (the one we entered through)
                    var normal = SIMD3<Float>(0, 0, 0)
                    var maxComponent: Float = 0
                    
                    // Check each axis
                    if abs(relativePos.x) > maxComponent {
                        maxComponent = abs(relativePos.x)
                        normal = SIMD3<Float>(relativePos.x > 0 ? 1 : -1, 0, 0)
                    }
                    if abs(relativePos.y) > maxComponent {
                        maxComponent = abs(relativePos.y)
                        normal = SIMD3<Float>(0, relativePos.y > 0 ? 1 : -1, 0)
                    }
                    if abs(relativePos.z) > maxComponent {
                        maxComponent = abs(relativePos.z)
                        normal = SIMD3<Float>(0, 0, relativePos.z > 0 ? 1 : -1)
                    }
                    
                    closestHit = RayVoxelIntersection(
                        hit: true,
                        voxelPosition: SIMD3<Int32>(voxelX, voxelY, voxelZ),
                        distance: currentDistance,
                        normal: normal
                    )
                    break
                }
            }
            
            currentDistance += stepSize
        }
        
        return closestHit
    }
    
    // MARK: - Voxel Editing
    
    func addVoxel(at position: SIMD3<Int32>, color: SIMD4<Float>) {
        guard position.x >= 0 && position.x < gridSize &&
              position.y >= 0 && position.y < gridSize &&
              position.z >= 0 && position.z < gridSize else { return }
        
        let index = Int(position.z * Int32(gridSize * gridSize) + position.y * Int32(gridSize) + position.x)
        
        guard index < voxelData.count else { return }
        
        voxelData[index].active = 1
        voxelData[index].color = color
        
        needsRebuild = true
    }
    
    func removeVoxel(at position: SIMD3<Int32>) {
        guard position.x >= 0 && position.x < gridSize &&
              position.y >= 0 && position.y < gridSize &&
              position.z >= 0 && position.z < gridSize else { return }
        
        let index = Int(position.z * Int32(gridSize * gridSize) + position.y * Int32(gridSize) + position.x)
        
        guard index < voxelData.count else { return }
        
        voxelData[index].active = 0
        
        needsRebuild = true
    }
    
    func clearAllVoxels() {
        for i in 0..<voxelData.count {
            voxelData[i].active = 0
        }
        voxelData.removeAll()
        needsRebuild = true
    }
    
    func handleTap(at point: CGPoint, viewSize: CGSize) {
        guard let editController = editController else { return }
        
        let ray = createRayFromScreenPoint(point, viewSize: viewSize)
        let intersection = raycastVoxels(ray)
        
        if intersection.hit {
            print("=== TAP DEBUG ===")
            print("Ray origin: \(ray.origin)")
            print("Ray direction: \(ray.direction)")
            print("Hit voxel: \(intersection.voxelPosition)")
            print("Hit normal: \(intersection.normal)")
            print("Distance: \(intersection.distance)")
            
            switch editController.editMode {
            case .remove:
                removeVoxel(at: intersection.voxelPosition)
                print("Removed voxel at \(intersection.voxelPosition)")
                
            case .add:
                let adjacentPos = SIMD3<Int32>(
                    intersection.voxelPosition.x + Int32(intersection.normal.x),
                    intersection.voxelPosition.y + Int32(intersection.normal.y),
                    intersection.voxelPosition.z + Int32(intersection.normal.z)
                )
                
                print("Adjacent position: \(adjacentPos)")
                
                // Bounds check
                if adjacentPos.x >= 0 && adjacentPos.x < gridSize &&
                   adjacentPos.y >= 0 && adjacentPos.y < gridSize &&
                   adjacentPos.z >= 0 && adjacentPos.z < gridSize {
                    
                    let color = editController.getColorComponents()
                    addVoxel(at: adjacentPos, color: color)
                    print("Added voxel at \(adjacentPos) with color \(color)")
                } else {
                    print("⚠️ Adjacent position out of bounds: \(adjacentPos)")
                }
            }
            print("=================\n")
        } else {
            print("No voxel hit - ray origin: \(ray.origin), direction: \(ray.direction)")
        }
    }

    // MARK: - Geometry Generation (original methods with minor modifications)
    
    func createThickAxisLines(thickness: Float = 0.015) {
        var vertices: [Vertex] = []
        var indices: [UInt32] = []
        let axisSize: Float = 1.0
        
        func addAxisPrism(start: SIMD3<Float>, end: SIMD3<Float>, color: SIMD4<Float>) {
            let direction = end - start
            let normalizedDir = normalize(direction)
            
            let isXAxis = abs(normalizedDir.x) > 0.9
            let isYAxis = abs(normalizedDir.y) > 0.9
            
            var corners: [SIMD3<Float>] = []
            
            if isXAxis {
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
            } else {
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
            
            for corner in corners {
                vertices.append(Vertex(position: corner, normal: normalizedDir, color: color))
            }
            
            let faceIndices: [UInt32] = [
                0, 1, 2,  0, 2, 3,
                4, 6, 5,  4, 7, 6,
                0, 4, 5,  0, 5, 1,
                2, 6, 7,  2, 7, 3,
                0, 3, 7,  0, 7, 4,
                1, 5, 6,  1, 6, 2
            ]
            
            for idx in faceIndices {
                indices.append(baseIdx + idx)
            }
        }
        
        addAxisPrism(
            start: SIMD3<Float>(0, 0, 0),
            end: SIMD3<Float>(axisSize, 0, 0),
            color: SIMD4<Float>(1, 0, 0, 1)
        )
        
        addAxisPrism(
            start: SIMD3<Float>(0, 0, 0),
            end: SIMD3<Float>(0, axisSize, 0),
            color: SIMD4<Float>(0, 1, 0, 1)
        )
        
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
    }
    
    func createGridLines() {
        let gridSizeCount = 64
        let spacing: Float = 0.2
        
        var vertices: [LineVertex] = []
        var indices: [UInt16] = []
        
        let halfSize = Float(gridSizeCount) * spacing / 2
        var currentIndex: UInt16 = 0
        let lineColor = SIMD4<Float>(0.3, 0.3, 0.3, 0.3)
        
        for i in 0...gridSizeCount {
            let position = Float(i) * spacing - halfSize
            
            vertices.append(LineVertex(position: SIMD3<Float>(position, 0, -halfSize), color: lineColor))
            vertices.append(LineVertex(position: SIMD3<Float>(position, 0, halfSize), color: lineColor))
            indices.append(currentIndex)
            indices.append(currentIndex + 1)
            currentIndex += 2
            
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
    }
    
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
    
    func calculateChunkBounds(voxelIndices: [UInt32]) -> AABB {
        guard !voxelIndices.isEmpty else {
            return AABB(min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(0, 0, 0))
        }
        
        let spacing = voxelSize * 1.1
        let halfGrid = Float(gridSize - 1) / 2.0
        
        var minPos = SIMD3<Float>(Float.infinity, Float.infinity, Float.infinity)
        var maxPos = SIMD3<Float>(-Float.infinity, -Float.infinity, -Float.infinity)
        
        for voxelIndex in voxelIndices {
            let voxel = voxelData[Int(voxelIndex)]
            let pos = voxel.position
            
            let worldPos = SIMD3<Float>(
                (Float(pos.x) - halfGrid) * spacing,
                (Float(pos.y) - halfGrid) * spacing,
                (Float(pos.z) - halfGrid) * spacing
            )
            
            let halfVoxel = voxelSize * 0.5
            minPos = simd_min(minPos, worldPos - SIMD3<Float>(repeating: halfVoxel))
            maxPos = simd_max(maxPos, worldPos + SIMD3<Float>(repeating: halfVoxel))
        }
        
        return AABB(min: minPos, max: maxPos)
    }
    
    func isChunkVisible(chunkIndex: Int, frustum: Frustum, cameraPosition: SIMD3<Float>) -> Bool {
        guard chunkIndex < chunkBounds.count else { return true }
        
        let bounds = chunkBounds[chunkIndex]
        let distanceToCenter = simd_length(cameraPosition - bounds.center)
        if distanceToCenter < 2.0 {
            return true
        }
        
        return frustum.intersects(aabb: bounds)
    }
    
    func generateVoxelMesh() {
        let voxelCount = gridSize * gridSize * gridSize
        
        print("=== Mesh Generation Start ===")
        print("Total voxels: \(voxelCount)")
        
        var params = VoxelParams(voxelSize: voxelSize, gridSize: Int32(gridSize))
        
        // Initialize or use existing voxel data
        if voxelData.isEmpty {
            voxelData = createDummyVoxels(gridSize: gridSize)
        }
        
        let activeCount = voxelData.filter { $0.active == 1 }.count
        print("Active voxels: \(activeCount)")
        
        // Update voxel buffer (reuse if same size, otherwise recreate)
        if voxelBuffer == nil || voxelBuffer.length != MemoryLayout<Voxel>.stride * voxelData.count {
            voxelBuffer = device.makeBuffer(
                bytes: voxelData,
                length: MemoryLayout<Voxel>.stride * voxelData.count,
                options: .storageModeShared)
        } else {
            // Reuse buffer - just update contents
            memcpy(voxelBuffer.contents(), voxelData, MemoryLayout<Voxel>.stride * voxelData.count)
        }
        
        // Reuse params buffer
        if paramsBuffer == nil {
            paramsBuffer = device.makeBuffer(
                bytes: &params,
                length: MemoryLayout<VoxelParams>.stride,
                options: .storageModeShared)
        } else {
            memcpy(paramsBuffer.contents(), &params, MemoryLayout<VoxelParams>.stride)
        }
        
        // Reuse compaction buffers (allocate once)
        if activeVoxelBuffer == nil {
            activeVoxelBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * voxelCount,
                options: .storageModeShared)
        }
        
        if activeCountBuffer == nil {
            activeCountBuffer = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
        }
        
        // Reset count to zero
        let countPtr = activeCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        countPtr.pointee = 0
        
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
        
        let compactedCount = Int(countPtr.pointee)
        print("Compacted to \(compactedCount) active voxels")
        
        let numChunks = (compactedCount + maxVoxelsPerChunk - 1) / maxVoxelsPerChunk
        print("Splitting into \(numChunks) chunk(s)")
        
        // Clear old chunk buffers
        vertexBuffers.removeAll()
        indexBuffers.removeAll()
        indexCounts.removeAll()
        chunkOffsets.removeAll()
        chunkBounds.removeAll()
        totalIndexCount = 0
        
        // Generate new chunks
        for chunkIndex in 0..<numChunks {
            let startVoxel = chunkIndex * maxVoxelsPerChunk
            let endVoxel = min(startVoxel + maxVoxelsPerChunk, compactedCount)
            let chunkSize = endVoxel - startVoxel
            
            let vertexBufferSize = MemoryLayout<Vertex>.stride * chunkSize * 24
            let indexBufferSize = MemoryLayout<UInt32>.stride * chunkSize * 36
            
            guard let vertexBuffer = device.makeBuffer(
                length: vertexBufferSize,
                options: .storageModePrivate),
                  let indexBuffer = device.makeBuffer(
                length: indexBufferSize,
                options: .storageModePrivate) else {
                continue
            }
            
            guard let commandBuffer2 = commandQueue.makeCommandBuffer() else { continue }
            var chunkActiveVoxels = Array<UInt32>()
            
            if let geometryEncoder = commandBuffer2.makeComputeCommandEncoder() {
                geometryEncoder.setComputePipelineState(geometryPipeline)
                geometryEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
                geometryEncoder.setBuffer(indexBuffer, offset: 0, index: 1)
                geometryEncoder.setBuffer(voxelBuffer, offset: 0, index: 2)
                geometryEncoder.setBuffer(paramsBuffer, offset: 0, index: 3)
                
                let activeVoxelPtr = activeVoxelBuffer.contents().bindMemory(to: UInt32.self, capacity: compactedCount)
                chunkActiveVoxels = Array(UnsafeBufferPointer(start: activeVoxelPtr.advanced(by: startVoxel), count: chunkSize))
                
                guard let chunkActiveBuffer = device.makeBuffer(
                    bytes: chunkActiveVoxels,
                    length: MemoryLayout<UInt32>.stride * chunkSize,
                    options: .storageModeShared) else { continue }
                
                var chunkCount = UInt32(chunkSize)
                guard let chunkCountBuffer = device.makeBuffer(
                    bytes: &chunkCount,
                    length: MemoryLayout<UInt32>.stride,
                    options: .storageModeShared) else { continue }
                
                geometryEncoder.setBuffer(chunkActiveBuffer, offset: 0, index: 4)
                geometryEncoder.setBuffer(chunkCountBuffer, offset: 0, index: 5)
                
                let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
                let threadgroups = MTLSize(width: (chunkSize + 63) / 64, height: 1, depth: 1)
                geometryEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
                geometryEncoder.endEncoding()
            }
            
            commandBuffer2.commit()
            commandBuffer2.waitUntilCompleted()
            
            let chunkIndexCount = chunkSize * 36
            let bounds = calculateChunkBounds(voxelIndices: chunkActiveVoxels)
            
            vertexBuffers.append(vertexBuffer)
            indexBuffers.append(indexBuffer)
            indexCounts.append(chunkIndexCount)
            chunkOffsets.append(startVoxel)
            chunkBounds.append(bounds)
            totalIndexCount += chunkIndexCount
        }
        
        print("\n=== Mesh Generation Complete ===")
        print("Total chunks: \(numChunks)")
        print("Total indices: \(totalIndexCount)")
        print("====================================\n")
        
        needsRebuild = false
    }
    
    func draw(in view: MTKView) {
        // Handle edit controller actions
        if let editController = editController {
            if editController.clearAll {
                clearAllVoxels()
                editController.clearAll = false
            }
        }
        
        // Rebuild mesh if needed
        if needsRebuild {
            generateVoxelMesh()
        }
        
        guard totalIndexCount > 0 else { return }
        
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
        
        let frustum = Frustum(viewProjectionMatrix: viewProj)
        
        var uniforms = Uniforms(viewProjectionMatrix: viewProj)
        let uniformBuffer = device.makeBuffer(bytes: &uniforms,
                                             length: MemoryLayout<Uniforms>.stride,
                                             options: [])
        
        encoder.setRenderPipelineState(renderPipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setCullMode(.back)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        
        for i in 0..<vertexBuffers.count {
            let shouldRender = !enableFrustumCulling || isChunkVisible(chunkIndex: i, frustum: frustum, cameraPosition: eye)
            
            if shouldRender {
                encoder.setVertexBuffer(vertexBuffers[i], offset: 0, index: 0)
                encoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: indexCounts[i],
                    indexType: .uint32,
                    indexBuffer: indexBuffers[i],
                    indexBufferOffset: 0)
            }
        }
        
        encoder.setRenderPipelineState(linePipeline)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(gridVertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        
        encoder.drawIndexedPrimitives(
            type: .line,
            indexCount: gridIndexCount,
            indexType: .uint16,
            indexBuffer: gridIndexBuffer,
            indexBufferOffset: 0)
        
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
        
        // Add tap gesture for voxel editing
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        view.addGestureRecognizer(tapGesture)
    }
    
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let view = gesture.view else { return }
        let location = gesture.location(in: view)
        handleTap(at: location, viewSize: view.bounds.size)
    }
    
    @objc private func handleOneFingerPan(_ gesture: UIPanGestureRecognizer) {
        if gesture.state == .began {
            lastPanLocation = gesture.location(in: gesture.view)
        } else if gesture.state == .changed {
            guard let last = lastPanLocation else { return }
            let currentLocation = gesture.location(in: gesture.view)

            let deltaX = Float(currentLocation.x - last.x) * 0.01
            let deltaY = Float(currentLocation.y - last.y) * 0.01
            let clampedDistance = max(cameraDistance, 0.5)
            let rotSpeed: Float = 0.5 * (clampedDistance / 3.0)
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
            let clampedDistance = max(cameraDistance, 0.5)
            let panSpeed: Float = 0.002 * clampedDistance
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
