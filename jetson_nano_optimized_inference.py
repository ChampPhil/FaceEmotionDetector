import os
import cv2
import numpy as np
import sys

from jetson_inference import detectNet
import jetson_utils
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy, cudaAllocMapped, cudaConvertColor

import time

try:
    os.remove('output.avi')
except Exception as e:
    pass

print(cv2.__version__)

cap = cv2.VideoCapture('/jetson-inference/ferplus/input_vid2.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('/jetson-inference/ferplus/output.avi', fourcc, 30, (1280, 720))




"""
net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                 threshold=args.threshold)
"""


net = detectNet("facedetect", sys.argv, 0.5)


def identify_faces(img):
    bgr_img = cudaFromNumpy(img, isBGR=True)
    rgb_img = cudaAllocMapped(width=736,
                          height=416,
						  format='rgb8')
    cudaConvertColor(bgr_img, rgb_img)


    #faces_rect = net.Detect(rgb_img, overlay="box,labels,conf")
    faces_rect = net.Detect(rgb_img, 736, 416, overlay="none")

    results = []
    for detection in faces_rect:
        print(detection)
        results.append([int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)])
    
    return results


print("\n\n\n\n\n\n\nBEGINNING DETECTION")
s = time.time()
while True:
    ret, img = cap.read()  # captures frame and returns boolean value and captured image
    
    if not ret:
        break
  
    resized_img = cv2.resize(img, (736, 416))
    
    
    """
    for detection in faces_rect:
        print("--------------------------------------")
        print(detection)
        print(type(detection))
        print(detection.Left)
        print("--------------------------------------")

    sys.exit()
    """

    

    
    for (x, y, x2, y2) in identify_faces(resized_img): #For each face
        #Drawing the rectangle around the face
        cv2.rectangle(resized_img, (x, y), (x2, y2), (0, 255, 0))
       
        
    out.write(cv2.resize(resized_img, (1280, 720)))
   
e = time.time()
print("Finished")
print(e - s)
cap.release()
out.release()
