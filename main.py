import os
import time
import numpy as np
import sys
from argparse import ArgumentParser

import torch
from torchvision import transforms
import tensorrt as trt
from onnx_helper import ONNXClassifierWrapper

import cv2
from jetson_inference import detectNet
import jetson_utils
from jetson_utils import videoSource, videoOutput, Log, cudaFromNumpy, cudaAllocMapped, cudaConvertColor


parser = ArgumentParser()
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--input", type=str, required=False)

args = parser.parse_args()

net = detectNet("facedetect", sys.argv, 0.5)

BATCH_SIZE = 1
N_CLASSES = 8 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("16workspace_size_64ms_inference.trt", [BATCH_SIZE, N_CLASSES])

transform_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
     
    ])

#Classnames to convert the model output to human-readable labels
classnames = ['neutral', 
                 'happiness', 
                 'surprise', 
                 'sadness', 
                 'anger', 
                 'disgust', 
                 'fear', 
                 'contempt'
]

def perform_inference(input_face_region):
   enlarged_region = transform_compose(input_face_region).unsqueeze(dim=0).numpy()

   #Code to predict an example with model here.....
   return trt_model.predict(enlarged_region)


  
def identify_faces(img):
    bgr_img = cudaFromNumpy(img, isBGR=True)
    rgb_img = cudaAllocMapped(width=bgr_img.width,
                          height=bgr_img.height,
						  format='rgb8')
    cudaConvertColor(bgr_img, rgb_img)


    faces_rect = net.Detect(rgb_img, overlay="box,labels,conf")

    results = []
    for detection in faces_rect:
        results.append([int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)])
    
    return results


def create_output_dir():
    if os.path.isdir('analyzed_videos') == False:
        print("CREATING DIRECTORY TO STORE ANALYZED VIDEOS")
        os.mkdir('analyzed_videos')

create_output_dir()


def create_intermediate_vid_name(directory_path):
    current_analyzed_files = len(os.listdir(directory_path))
    return f"input_vid{current_analyzed_files+1}.avi"

if args.input == None:
    input_vidname = create_intermediate_vid_name('analyzed_videos')
else:
    input_vidname = args.input

if os.path.exists(os.path.join('analyzed_videos', args.output)):
    print("The output vidname you specified already exists!!")
    print("Please rerun this script with a unique output video name.")
    print("\n\nExiting...")
    sys.exit()

if input_vidname == None:
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))


    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(input_vidname, fourcc, 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(cv2.resize(frame, (frame_width, frame_height)))
            cv2.imshow('Livestream Video Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()




def create_output_vid(pathname, output_vid):
    cap = cv2.VideoCapture(pathname)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(output_vid, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            break
        
        #Detecting the faces
    
        for (x, y, x2, y2) in identify_faces(img): #For each face
            #Drawing the rectangle around the face
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0)) 
            output_logit = perform_inference(img[y:y2, x:x2])
            pred = torch.argmax(torch.softmax(torch.from_numpy(output_logit).squeeze(), dim=0), dim=0)
            
            cv2.putText(img, classnames[pred], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
        out.write(cv2.resize(img, (frame_width, frame_height)))
   

    cap.release()
    out.release()

def delete_video(vidname):
    if os.path.exists(vidname):
        os.remove(vidname)

if args.input == None:
    create_output_vid(os.path.join('analyzed_videos', input_vidname), 
                    os.path.join('analyzed_videos', args.output))
else:
    create_output_vid(input_vidname, 
    os.path.join('analyzed_videos', args.output))

if args.input == None:
    delete_video(input_vidname)




