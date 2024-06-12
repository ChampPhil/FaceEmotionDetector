#Imports
import create_iterable_dataset
import torch
from torchvision import transforms
import tensorrt as trt
import torch
from onnx_helper import ONNXClassifierWrapper
import cv2
import time

#Transforms to convert raw images to inputs that can be understood by the model
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

#Creating the engine
BATCH_SIZE = 1
N_CLASSES = 8 # Our ResNet-50 is trained on a 1000 class ImageNet task
ENGINE_NAME = "16workspace_size_64ms_inference.trt"
trt_model = ONNXClassifierWrapper(ENGINE_NAME, [BATCH_SIZE, N_CLASSES])

pathname = 'input_vid2.mp4'
output_name = 'output.avi'

cap = cv2.VideoCapture(pathname)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

def perform_inference(input_face_region):
   enlarged_region = transform_compose(input_face_region).unsqueeze(dim=0).numpy()
   #Code to predict an example with model here.....

   return trt_model.predict(enlarged_region)

   #Code to predict an example with model here
  

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def identify_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rect = face_haar_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.3, 
        minNeighbors=9,
    ) 

    return faces_rect

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(output_name, fourcc, fps, (frame_width, frame_height))

print("Starting video creation/inference")
s = time.time()
while True:
    ret, img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        break
    
    #Detecting the faces
    

    for (x, y, w, h) in identify_faces(img): #For each face
        #Drawing the rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 
        output_logit = perform_inference(img[y:y+h, x:x+w])
        pred = torch.argmax(torch.softmax(torch.from_numpy(output_logit).squeeze(), dim=0), dim=0)
        
        cv2.putText(img, classnames[pred], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    out.write(cv2.resize(img, (frame_width, frame_height)))
   
e = time.time()
print("Finished")
print(e - s)
cap.release()
out.release()
