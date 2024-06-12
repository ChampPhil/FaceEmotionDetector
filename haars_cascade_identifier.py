import os
import cv2
import numpy as np

print(cv2.__version__)
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture("/jetson-inference/ferplus/input_vid1.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('/jetson-inference/ferplus/output.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        break
    
    #Detecting the faces
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rect = face_haar_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.3, 
        minNeighbors=3,
    ) 

    for (x, y, w, h) in faces_rect: #For each face
        #Drawing the rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 
        
        #Resizing the region of interest to (224, 224)
        # cropping region of interest i.e. face area from  image
       
    
        #Performing preprocessing on the image so it can be passed into the model
        
        #Converting it into a Keras-readable image tensor
        #Giving it a batch size of 1
        #Normalizing values to 0-1. 

        #Running a forward-pass
        #...

        # find the index of the output label 
        #...

        #Converting output to a label
        #...
        
        #Playing label.
        #cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    out.write(cv2.resize(img, (frame_width, frame_height)))
   

print("Finished")
cap.release()
out.release()
