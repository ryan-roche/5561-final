//
//  ContentView.swift
//  taco-ios
//
//  Created by Ryan Roche on 12/8/24.
//

import AVFoundation
import CoreML
import Vision
import SwiftUI

struct Detection {
    let boundingBox: CGRect
    let label: String
    let confidence: Float
}

struct ContentView: View {
    @State private var currentModel: ModelOption = .taco
    @State private var lastInferenceTime: TimeInterval = CACurrentMediaTime()
    @State private var fps: Double = 0

    var body: some View {
        CameraView { frame in
            // Whenever we receive a frame from the CameraView, perform inference on it using the selected model and return the results so that CameraView can display an annotated image

            // Ensure model and buffer are available
            guard let model = currentModel.model,
                let pixelBuffer = CMSampleBufferGetImageBuffer(frame)
            else {
                return []
            }
            
            // Update FPS
            let currentTime = CACurrentMediaTime()
            let timeElapsed = currentTime - lastInferenceTime
            fps = 1.0 / timeElapsed
            lastInferenceTime = currentTime

            // Perform inference with CoreML
            do {
                let visionModel = try VNCoreMLModel(for: model)
                let request = VNCoreMLRequest(model: visionModel)
                try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
                
                // Now that inference is complete, we can safely access results
                guard let results = request.results as? [VNRecognizedObjectObservation] else {
                    print("Got no results")
                    return []
                }
                
                print("Detected \(results.count) objects")
                
                return results.map { observation in
                    Detection(
                        boundingBox: observation.boundingBox,
                        label: observation.labels[0].identifier,
                        confidence: observation.confidence
                    )
                }
                
            } catch {
                print("Failed to perform inference: \(error)")
                return []
            }
        }
        .ignoresSafeArea(.all)
        
        // MARK: FPS overlay
        .overlay(alignment: .topLeading) {
            Text(String(format: "FPS: %.1f", fps))
                .font(.system(.body, design: .monospaced))
                .padding(8)
                .background {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.white)
                        .shadow(color:.black.opacity(0.2), radius: 4, y:2)
                }
                .foregroundColor(.black)
                .padding(.top, 44)
                .padding(.leading)
        }
        
        // MARK: Model Selector
        .overlay(alignment: .bottom) {
            Picker("Model", selection: $currentModel) {
                ForEach(ModelOption.allCases) { option in
                    Text(option.name)
                        .tag(option)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
        }
    }
}

#Preview {
    ContentView()
}
