//
//  CameraViewController.swift
//  FaceCreeper
//
//  Created by Austin Han on 2017-01-21.
//  Copyright © 2017 Hackathon. All rights reserved.
//

import UIKit
import AVFoundation

class CameraViewController: UIViewController, AVCapturePhotoCaptureDelegate{

    @IBOutlet weak var previewView: UIView!
    var session: AVCaptureSession?
    var stillImageOutput: AVCapturePhotoOutput?
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?

    @IBOutlet weak var previewImageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        session = AVCaptureSession()
        session!.sessionPreset = AVCaptureSessionPresetPhoto
        let backCamera = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo)
        
        var error: NSError?
        var input: AVCaptureDeviceInput!
        do {
            input = try AVCaptureDeviceInput(device: backCamera)
        } catch let error1 as NSError {
            error = error1
            input = nil
            print(error!.localizedDescription)
        }
        
        if error == nil && session!.canAddInput(input) {
            session!.addInput(input)
            // ...
            // The remainder of the session setup will go here...
        }
        
        stillImageOutput = AVCapturePhotoOutput()
        if session!.canAddOutput(stillImageOutput) {
            session!.addOutput(stillImageOutput)
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: session)
            videoPreviewLayer!.videoGravity = AVLayerVideoGravityResizeAspect
            videoPreviewLayer!.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
            previewView.layer.addSublayer(videoPreviewLayer!)
            session!.startRunning()
        }
        
        
        
        
//        stillImageOutput?.

    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        videoPreviewLayer!.frame = previewView.bounds
        
        
    }
    func capture(_ captureOutput: AVCapturePhotoOutput, didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?, previewPhotoSampleBuffer: CMSampleBuffer?, resolvedSettings: AVCaptureResolvedPhotoSettings, bracketSettings: AVCaptureBracketedStillImageSettings?, error: Error?) {
        
        
        if photoSampleBuffer != nil {
            let imageData = AVCaptureStillImageOutput.jpegStillImageNSDataRepresentation(photoSampleBuffer)
            
            let dataProvider = CGDataProvider(data: imageData as! CFData)
            let cgImageRef = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: CGColorRenderingIntent.defaultIntent)
            let image = UIImage(cgImage: cgImageRef!, scale: 1.0, orientation: UIImageOrientation.right)
            checkImage(image: image)
            
            print("SUCCESS")
            // ...
            // Add the image to captureImageView here...
        }
    }
    
    
    @IBAction func CaptureButton(_ sender: Any) {
        
        if let videoConnection = stillImageOutput!.connection(withMediaType: AVMediaTypeVideo) {
            
//            for i in 1...10{
            let settings = AVCapturePhotoSettings()
            let previewPixelType = settings.availablePreviewPhotoPixelFormatTypes.first!
            let previewFormat = [kCVPixelBufferPixelFormatTypeKey as String: previewPixelType,
                                 kCVPixelBufferWidthKey as String: 160,
                                 kCVPixelBufferHeightKey as String: 160]
            settings.previewPhotoFormat = previewFormat
            
            stillImageOutput?.capturePhoto(with: settings, delegate: self)

//            }
            
        }

    }
    
    func  checkImage(image:UIImage) {

        guard let detectedImage = CIImage(image: image) else {
            return
        }
        
        let detectionAccuracy = [CIDetectorAccuracy: CIDetectorAccuracyHigh]
        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: detectionAccuracy)
        let foundFaces = faceDetector?.features(in: detectedImage, options: [CIDetectorImageOrientation: 5])
        
        
        // For converting the Core Image Coordinates to UIView Coordinates
//        let detectedImageSize = detectedImage.extent.size
//        var transform = CGAffineTransform(scaleX: 1, y: 1)
//        transform = transform.translatedBy(x: 0, y: detectedImageSize.height)
        
        let detectedImageSize = detectedImage.extent.size
        var transform = CGAffineTransform(scaleX: 1, y: -1)
        transform = transform.translatedBy(x: 0, y: -detectedImageSize.height)
        

        for face in foundFaces as! [CIFaceFeature] {
            print("Found bounds are \(face.bounds)")
//        
//            let cgImage = image.cgImage
//            let croppedCGImage: CGImage = cgImage!.cropping(to: face.bounds)!
//            let uiImage = UIImage(cgImage: croppedCGImage, scale: 1, orientation: image.imageOrientation)
//            previewImageView.image = uiImage;
//            UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil);
            
            
            
            var newBounds = face.bounds.applying(transform)
            
            // Calculate the actual position of indicator
            let viewSize = previewImageView.bounds.size
            let scale = min(viewSize.width / detectedImageSize.width,
                            viewSize.height / detectedImageSize.height)
            let offsetX = (viewSize.width - detectedImageSize.width * scale) / 2
            let offsetY = (viewSize.height - detectedImageSize.height * scale) / 2
            
            newBounds = newBounds.applying(CGAffineTransform(scaleX: scale, y: scale))
            newBounds.origin.x += offsetX
            newBounds.origin.y += offsetY
            
            let indicator = UIView(frame: newBounds)
            indicator.backgroundColor = UIColor.clear
            indicator.layer.borderWidth = 2
            indicator.layer.borderColor = UIColor.red.cgColor
            previewImageView.addSubview(indicator)
            
        }
    }
    
    func UploadRequest(image:UIImage)
    {
        let url = NSURL(string: "http://www.kaleidosblog.com/tutorial/upload.php")
        
        let request = NSMutableURLRequest(url: url! as URL)
        request.httpMethod = "POST"
        
        let boundary = generateBoundaryString()
        
        //define the multipart request type
        
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        
        let image_data = UIImagePNGRepresentation(image)
    
        
        let body = NSMutableData()
        
        let fname = "test.png"
        let mimetype = "image/png"
        
        //define the data post parameter
        
        body.append("--\(boundary)\r\n".data(using: String.Encoding.utf8)!)
        body.append("Content-Disposition:form-data; name=\"test\"\r\n\r\n".data(using: String.Encoding.utf8)!)
        body.append("hi\r\n".data(using: String.Encoding.utf8)!)
        
        
        
        body.append("--\(boundary)\r\n".data(using: String.Encoding.utf8)!)
        body.append("Content-Disposition:form-data; name=\"file\"; filename=\"\(fname)\"\r\n".data(using: String.Encoding.utf8)!)
        body.append("Content-Type: \(mimetype)\r\n\r\n".data(using: String.Encoding.utf8)!)
        body.append(image_data!)
        body.append("\r\n".data(using: String.Encoding.utf8)!)
        
        
        body.append("--\(boundary)--\r\n".data(using: String.Encoding.utf8)!)
        
        
        
        request.httpBody = body as Data
        
        
        
        let session = URLSession.shared
        
        
        let task = session.dataTask(with: request as URLRequest) {
            (
            data, response, error) in
            
            guard let _:NSData = data as NSData?, let _:URLResponse = response, error == nil else {
                print("error")
                return
            }
            
            let dataString = NSString(data: data!, encoding: String.Encoding.utf8.rawValue)
            
            print(dataString)
            
        }
        
        task.resume()
        
        
    }
    
    
    func generateBoundaryString() -> String
    {
        return "Boundary-\(NSUUID().uuidString)"
    }
    

}
