import SwiftUI
import Combine

enum EditMode {
    case add
    case remove
}

class VoxelEditController: ObservableObject {
    @Published var editMode: EditMode = .add
    @Published var selectedColor: Color = .blue
    @Published var clearAll: Bool = false
    @Published var tapLocation: CGPoint?
    
    func toggleMode() {
        editMode = (editMode == .add) ? .remove : .add
    }
    
    func getColorComponents() -> SIMD4<Float> {
        let uiColor = UIColor(selectedColor)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        uiColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        
        return SIMD4<Float>(Float(red), Float(green), Float(blue), Float(alpha))
    }
}
