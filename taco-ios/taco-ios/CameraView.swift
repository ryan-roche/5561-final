//
//  CameraView.swift
//  taco-ios
//
//  Created by Ryan Roche on 12/10/24.
//


import SwiftUI
import AVFoundation

struct CameraView: UIViewControllerRepresentable {
    static let imageWidth: CGFloat = 4032
    static let imageHeight: CGFloat = 3024
    
    class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        var parent: CameraView
        weak var previewLayer: AVCaptureVideoPreviewLayer?
        
        init(_ parent: CameraView) {
            self.parent = parent
        }
        
        func captureOutput(_ output: AVCaptureOutput,
                           didOutput sampleBuffer: CMSampleBuffer,
                           from connection: AVCaptureConnection) {
            let detections = parent.processFrame(sampleBuffer)
            
            DispatchQueue.main.async {
                self.previewLayer?.sublayers?.removeAll(where: { $0 is CAShapeLayer || $0 is CATextLayer })
                
                guard let previewLayer = self.previewLayer else { return }
                
                for detection in detections {
                    let shapeLayer = CAShapeLayer()
                    shapeLayer.fillColor = nil
                    shapeLayer.strokeColor = UIColor.green.cgColor
                    shapeLayer.lineWidth = 2
                    
                    // MARK: Transform coordinates of bounding box to camera frame
                    let scaledRect = CGRect(
                        x: detection.boundingBox.minX * CameraView.imageWidth,
                        y: (1 - detection.boundingBox.maxY) * CameraView.imageHeight,
                        width: detection.boundingBox.width * CameraView.imageWidth,
                        height: detection.boundingBox.height * CameraView.imageHeight
                    )

                    let normalizedRect = CGRect(
                        x: scaledRect.minX / CameraView.imageWidth,
                        y: scaledRect.minY / CameraView.imageHeight,
                        width: scaledRect.width / CameraView.imageWidth,
                        height: scaledRect.height / CameraView.imageHeight
                    )

                    // MARK: Draw bounding box with label and logit
                    let convertedRect = previewLayer.layerRectConverted(fromMetadataOutputRect: normalizedRect)
                    shapeLayer.path = UIBezierPath(rect: convertedRect).cgPath
                    
                    let textLayer = CATextLayer()
                    textLayer.string = "\(detection.label): \(String(format: "%.2f", detection.confidence))"
                    textLayer.fontSize = 12
                    textLayer.foregroundColor = UIColor.green.cgColor
                    textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
                    
                    let textHeight: CGFloat = 20
                    textLayer.frame = CGRect(
                        x: convertedRect.minX,
                        y: max(0, convertedRect.minY - textHeight),
                        width: 150,
                        height: textHeight
                    )
                    
                    previewLayer.addSublayer(shapeLayer)
                    previewLayer.addSublayer(textLayer)
                }
            }
        }
    }

    let processFrame: (CMSampleBuffer) -> [Detection]

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }

    func makeUIViewController(context: Context) -> UIViewController {
        let controller = UIViewController()
        let session = AVCaptureSession()
        session.sessionPreset = .photo
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return controller
        }
        
        session.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(context.coordinator, queue: DispatchQueue(label: "cameraFrameQueue"))
        session.addOutput(videoOutput)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = controller.view.bounds
        controller.view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        
        session.startRunning()
        
        return controller
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        context.coordinator.previewLayer?.frame = uiViewController.view.bounds
    }
}
