//
//  ContentView.swift
//  GPUVoxelRenderingTest
//
//  Created by Yasuo Hasegawa on 2025/06/05.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var editController = VoxelEditController()
    
    var body: some View {
        ZStack{
            MetalView(editController: editController)
            VStack {
                Spacer()
                
                // Edit mode controls
                HStack(spacing: 20) {
                    // Mode toggle
                    Button(action: {
                        editController.toggleMode()
                    }) {
                        HStack {
                            Image(systemName: editController.editMode == .add ? "plus.circle.fill" : "minus.circle.fill")
                            Text(editController.editMode == .add ? "Add Mode" : "Remove Mode")
                        }
                        .padding()
                        .background(editController.editMode == .add ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    
                    // Color picker (only visible in add mode)
                    if editController.editMode == .add {
                        ColorPicker("", selection: $editController.selectedColor)
                            .labelsHidden()
                            .scaleEffect(1.5)
                            .padding()
                            .background(Color.white.opacity(0.8))
                            .cornerRadius(10)
                    }
                    
                    // Clear all button
                    Button(action: {
                        editController.clearAll = true
                    }) {
                        Image(systemName: "trash.fill")
                            .padding()
                            .background(Color.orange.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
                .padding()
                .background(Color.black.opacity(0.3))
                .cornerRadius(15)
                .padding(.bottom, 30)
            }
            
//            // Crosshair for aiming
//            Circle()
//                .stroke(Color.white, lineWidth: 2)
//                .frame(width: 20, height: 20)
//                .shadow(color: .black, radius: 2)
        }
        .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
}
