//
//  taco_iosApp.swift
//  taco-ios
//
//  Created by Ryan Roche on 12/8/24.
//

import SwiftUI
import CoreML

enum ModelOption: CaseIterable, Identifiable {
    var id: String { self.name }
    
    case taco
    case yoloV3
    
    var name: String {
        switch self {
        case .taco:
            return "TACO-trained YOLOv8"
        case .yoloV3:
            return "YOLOv3 (for testing)"
        }
    }
    
    var model: MLModel? {
        switch self {
        case .taco:
            return try? taco_yolo(configuration: MLModelConfiguration()).model
        case .yoloV3:
            return try? YOLOv3FP16(configuration: MLModelConfiguration()).model
        }
    }
}

@main
struct taco_iosApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
