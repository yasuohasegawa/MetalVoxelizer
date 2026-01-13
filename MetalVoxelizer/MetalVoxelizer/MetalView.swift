//
//  MetalView.swift
//  MetalVoxelizer
//
//  Created by Yasuo Hasegawa on 2026/01/13.
//

import SwiftUI
import MetalKit

struct MetalView: UIViewRepresentable {
    @ObservedObject var editController: VoxelEditController
    
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

        let renderer = VoxelRenderer(mtkView: mtkView, editController: editController)
        context.coordinator.renderer = renderer
        
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}
