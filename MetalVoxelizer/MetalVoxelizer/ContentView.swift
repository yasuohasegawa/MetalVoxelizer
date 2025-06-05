//
//  ContentView.swift
//  GPUVoxelRenderingTest
//
//  Created by Yasuo Hasegawa on 2025/06/05.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack{
            MetalView()
        }
        .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
}
