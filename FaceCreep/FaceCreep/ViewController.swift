 //
//  ViewController.swift
//  FaceCreep
//
//  Created by Austin Han on 2017-01-20.
//  Copyright © 2017 Hackathon. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var ImageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
//        detect()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    @IBAction func takePicture(_ sender: Any) {
        
        let imagePickerController = UIImagePickerController()
        imagePickerController.sourceType = .photoLibrary
        imagePickerController.delegate = self
        present(imagePickerController, animated: true, completion: nil)

        
    }
//    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo     if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
//            imagePost.image = image
//        } else{
//            print("Something went wrong")
//        }
//        
//        self.dismiss(animated: true, completion: nil)
//
//        detect()
//    }
    
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : AnyObject]) {
        print("HELLO WORLD")
        if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
            ImageView.image = image
            detect(image:image)

        } else{
            print("Something went wrong")
        }
        
        self.dismiss(animated: true, completion: nil)
    }
    
    
    func detect(image: UIImage){
    
        guard let detectedImage = CIImage(image: image) else {
            return
        }
        
        let detectionAccuracy = [CIDetectorAccuracy: CIDetectorAccuracyHigh]
        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: detectionAccuracy)
        let foundFaces = faceDetector?.features(in: detectedImage)
        print(foundFaces)
        
        
        if let face = foundFaces?.first as? CIFaceFeature {
            print("Found face at \(face.bounds)")
            
            if face.hasLeftEyePosition {
                print("Found left eye at \(face.leftEyePosition)")
            }
            
            if face.hasRightEyePosition {
                print("Found right eye at \(face.rightEyePosition)")
            }
            
            if face.hasMouthPosition {
                print("Found mouth at \(face.mouthPosition)")
            }
        }
        
//
//        let accuracy = [CIDetectorAccuracy: CIDetectorAccuracyHigh]
//        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: accuracy)
//        let faces = faceDetector?.features(in: personciImage)
//        print("FACES")
//        print(faces)
//        // converting to other coordinate system
//        let ciImageSize = personciImage.extent.size
//        var transform = CGAffineTransform(scaleX: 1, y: -1)
//        transform = transform.translatedBy(x: 0, y: -ciImageSize.height)
//        
//        print("HELLO WORLD")
//        
//        
//        for face in faces as! [CIFaceFeature] {
//            
//            print("Found bounds are \(face.bounds)")
//            
//            // calculating place for faceBox
//            var faceViewBounds = face.bounds.applying(transform)
//            
//            let viewSize = ImageView.bounds.size
//            let scale = min(viewSize.width / ciImageSize.width,
//                            viewSize.height / ciImageSize.height)
//            let offsetX = (viewSize.width - ciImageSize.width * scale) / 2
//            let offsetY = (viewSize.height - ciImageSize.height * scale) / 2
//            
//            faceViewBounds = faceViewBounds.applying(CGAffineTransform(scaleX: scale, y: scale))
//            faceViewBounds.origin.x += offsetX
//            faceViewBounds.origin.y += offsetY
//            
//            let faceBox = UIView(frame: faceViewBounds)
//            
//            faceBox.layer.borderWidth = 3
//            faceBox.layer.borderColor = UIColor.red.cgColor
//            faceBox.backgroundColor = UIColor.clear
//            //            pImage.addSubview(faceBox)
//            
//            print(face.bounds)
//            
//            if face.hasLeftEyePosition {
//                print("Left eye bounds are \(face.leftEyePosition)")
//            }
//            
//            if face.hasRightEyePosition {
//                print("Right eye bounds are \(face.rightEyePosition)")
//            }
//        }
//        
//        dismiss(animated: true, completion: nil)
        

    }
    

    
    
//    down vote
//    iOS have native face detection in CoreImage framework that works pretty cool. You can as well detect eyes etc. Just check out this code, there's show how you can work with it.
//
//    func detect(personPic) {
//
//        //    }
//



}

