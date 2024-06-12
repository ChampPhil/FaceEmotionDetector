# Analyzing Facial Emotions Using a Deep Learning Pipeline

## Overview/Basic Understanding

Facial emotion analysis is an intriguing application of deep learning, combining the power of artificial intelligence with human emotion recognition. This article explains a Python-based project that processes raw video sequences to detect faces, analyze emotions, and generate an annotated output video. Our pipeline leverages several sophisticated tools and frameworks, including Haar cascades, a custom-trained neural network based on the VGG16 architecture, and the NVIDIA TensorRT framework for optimization.

<ins>The Pipeline Workflow is as Follows:</ins>

- *Raw Video Input*: Capture or load a raw video sequence containing human faces.
  
- *Face Detection*: Use Haar cascades to identify faces in each frame of the video.

- *Emotion Analysis*: Feed the preprocessed data (a.k.a the detected face) into a custom-trained Deep Neural Network (DNN) based on VGG16 to predict facial emotions.

- *Video Output Generation*: Use OpenCV to create a new video sequence that highlights detected faces and labels the emotions.

### Step 1: Raw Video Input

The first step is to acquire a raw video sequence. In our instance, this is done by prerecording a video with a seperate python script. Then, I used OpenCV for video handling, which allows easy frame extraction and manipulation.


### Step 2: Face Detection Using Haar Cascades

Haar cascades are a popular choice for real-time face detection. They work by training a cascade function on lots of positive and negative images, which is then used to detect objects in other images. The grayscale conversion of each frame is essential for accurate detection, followed by applying the Haar cascade classifier to identify faces.


### Step 3: Emotion Analysis with Custom-Trained DNN

We use a custom-trained neural network based on the VGG16 architecture to perform the emotion analysis. The network is optimized using the NVIDIA TensorRT framework to enhance inference speed and efficiency. This involves 

-  fine-tuning the VGG16 model after altering its structure to output emotion predictions
-  training this altered architecture
-  converting the model for TensorRT optimization.

### Step 4: Video Output Generation with OpenCV

Finally, the project generates a new video sequence using OpenCV. This new video highlights the detected faces and labels the emotions. Each frame is annotated with rectangles around detected faces and text labels for the predicted emotions.

### Conclusion

This project demonstrates a comprehensive pipeline for facial emotion analysis using deep learning. By integrating tools like Haar cascades, OpenCV, PyTorch, and TensorRT, we can efficiently process and analyze video data, providing meaningful insights into human emotions. The result is an enhanced video sequence that visually represents detected faces and their corresponding emotions, showcasing the power of AI in understanding human expressions.


## Usage