# Analyzing Facial Emotions Using a Deep Learning Pipeline

## Overview/Basic Understanding

Facial emotion analysis is an intriguing application of deep learning, combining the power of artificial intelligence with human emotion recognition. This article explains a Python-based project that processes raw video sequences to detect faces, analyze emotions, and generate an annotated output video. Our pipeline leverages several sophisticated tools and frameworks, including a TAO Detection Model, a custom-trained neural network based on the VGG16 architecture, and the NVIDIA TensorRT framework for optimization.

To eloborate, my project offers two main options:
   - Running facial emotion analysis on a preexisting video 

   - Create a video in realtime, and then run facial emotion analysis on that newly created video


<ins>The Pipeline Workflow is as Follows:</ins>

- *Raw Video Input*: Capture or load a raw video sequence containing human faces.
  
- *Face Detection*: Use a pretrained TAO Detection Model to identify faces in each frame of the video.

- *Emotion Analysis*: Feed the preprocessed data (a.k.a the detected face) into a custom-trained Deep Neural Network (DNN) based on VGG16 to predict facial emotions.

- *Video Output Generation*: Use OpenCV to create a new video sequence that highlights detected faces and labels the emotions.

### Step 1: Raw Video Input

The first step is to acquire a raw video sequence. In our instance, this is done by prerecording a video with a seperate python script. Then, I used OpenCV for video handling, which allows easy frame extraction and manipulation.


### Step 2: Face Detection Using Haar Cascades

Train Adapt Optimize (TAO) Toolkit is a AI toolkit for taking purpose-built pre-trained AI models (*like one that identfies faces*) and customizing them with your own data. TAO adapts popular network architectures and backbones to your data, allowing you to train, fine tune, prune and export highly optimized and accurate AI models for edge deployment (on hardware like the Jetson Nano). In my instance, I used the "face-detect" TAO Model.


### Step 3: Emotion Analysis with Custom-Trained DNN

We use a custom-trained neural network based on the VGG16 architecture to perform the emotion analysis. The network is optimized using the NVIDIA TensorRT framework to enhance inference speed and efficiency. This involves 

-  fine-tuning the VGG16 model after altering its structure to output emotion predictions
-  training this altered architecture
-  converting the model for TensorRT optimization.

### Step 4: Video Output Generation with OpenCV

Finally, the project generates a new video sequence using OpenCV. This new video highlights the detected faces and labels the emotions. Each frame is annotated with rectangles around detected faces and text labels for the predicted emotions.



## Requirements
  
-  Have the jetson-inference and jetson-utils libraries up and running

-  Enter "pip install -r requirements.txt" into the CLI to install the approriate versions of necessary libraries

## Using the Project from Jetson Nano CLI

### Run Inference on a Preexisting Video
   
   -  Run the following command: "python3 main.py --input="INPUT_VIDNAME" --output="OUTPUT_VIDNAME". 

      - **INPUT_VIDNAME** is the filepath to the preexisting video (you should put the video in the same directory as main.py so you only have to input the name of the video)
         -  *NOTE*: For this argument, include the file extension. So instead of saying "input_vid1", enter "input_vid1.*avi*"

      - **OUTPUT_VIDNAME** is what you want the name of the output video sequence (that visually represents detected faces and their corresponding emotions). The output video sequence will be created in a directory called 'analyzed_videos' that main.py automically generates. 
         -  *NOTE*: For this argument, include the file extension. So instead of saying "output_vid1", enter "output_vid1.*avi*"


### Run Inference on a Newly Created Video

   - *NOTE*: You can only do this if you have a monitor and keyboard setup
   
   -  Run the following command: "python3 main.py --output="OUTPUT_VIDNAME". 

## Conclusion 

This project demonstrates a comprehensive pipeline for facial emotion analysis using deep learning. By integrating tools like Haar cascades, OpenCV, PyTorch, and TensorRT, we can efficiently process and analyze video data, providing meaningful insights into human emotions. The result is an enhanced video sequence that visually represents detected faces and their corresponding emotions, showcasing the power of AI in understanding human expressions.


