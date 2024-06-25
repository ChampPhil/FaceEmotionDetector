import os
import cv2
import numpy as np
import sys

from jetson_inference import detectNet
import jetson_utils
from jetson_utils import videoSource, videoOutput, Log

print(cv2.__version__)


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('/jetson-inference/ferplus/output.avi', fourcc, 30, (1280, 720))




"""
net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                 threshold=args.threshold)
"""


net = detectNet("facedetect", sys.argv, 0.5)

input_vid = videoSource("file:///jetson-inference/ferplus/input_vid1.mp4")


print("\n\n\n\n\n\n\nBEGINNING DETECTION")
while True:
    img = input_vid.Capture()

    if img is None: # timeout
        continue  
  
    faces_rect = net.Detect(img, overlay="box,labels,conf")
   
    print("\n\n\nPrinting out each output")
    
    """
    for detection in faces_rect:
        print("--------------------------------------")
        print(detection)
        print(type(detection))
        print(detection.Left)
        print("--------------------------------------")

    sys.exit()
    """

    bgr_img = jetson_utils.cudaAllocMapped(width=width, height=height, format='bgr8')
    jetson_utils.cudaConvertColor(img, bgr_img)

    # Convert to numpy array
    cv_img = jetson_utils.cudaToNumpy(bgr_img)

    for detection in faces_rect: #For each face
        #Drawing the rectangle around the face
        cv2.rectangle(cv_img, (int(detection.Left), int(detection.Top)), (int(detection.Right), int(detection.Bottom)), (0, 255, 0), 2) 
        
    out.write(cv2.resize(cv_img, (1280, 720)))
   

print("Finished")
cap.release()
out.release()
