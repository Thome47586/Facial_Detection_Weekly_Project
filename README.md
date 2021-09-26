# Facial Detection Project
__Date:__ 26th of September 2021

__Authors:__ 

 + __Tuan Le__: https://github.com/tuanle163
 + __Duy__: https://github.com/yuthefirst 
 + __Thome__: https://github.com/Thome47586 
 + __Vu Tran Huy__: https://github.com/vutrhuy81
 
# I. Overview
This is small group project of four members. 

The project is to detect our team members face and name. We use the OpenCV, YOLO and Deep Learning model to train and detect our team member faces. 

<b>Model diagram</b>

# II. Requirement and Dataset

## Requirements
Following packages that you need to install:

+ python>=3.8
+ opencv-contrib-python
+ tensorflow
+ numpy
+ matplotlib

You can create an Anaconda environment using our following __.yml__ files:

+ __Macbook M1 (ARM64 CPU):__ 

```terminal
conda env create create -f ./env/M1_facial_detection.yml
```

+ __Macbook Intel CPU:__
```terminal
conda env create create -f ./env/MacIntel_facial_detection.yml
```
+ __Window Intel CPU:__
```terminal
conda env create create -f ./env/WinIntel_facial_detection.yml
```

## Dataset

The dataset we used to train our model can be found here: https://github.com/tuanle163/Facial_Detection_Weekly_Project/tree/main/dataset

Or you can download it from Google Drive link: [Link](https://drive.google.com/file/d/1NYiarTKMgdsrWHXcRhmVblyPY3kJmuvj/view?usp=sharing)

# III. Model Structure
![image](https://user-images.githubusercontent.com/29221802/134808537-bd1df0a8-5ebc-4f36-ad0b-13a1a2ebd38f.png)

## Image Preprocessing


## YOLOV3 Model
![image](https://user-images.githubusercontent.com/29221802/134807498-77eb1ed1-58ac-4cba-bdce-d1be91367c1f.png)
### What is YOLOV3?
YOLOV3 (You Only Look Once, Version 3) is real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. YOLO uses features learned by a deep convolutional neural network to detect an object. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi. 
The first version of YOLO was created in 2016, and version 3, which is discussed extensively in this article, was made two years later in 2018. YOLOV3 is an improved version of YOLO and YOLOV2. YOLO is implemented using the Keras or OpenCV deep learning libraries.
Object classification systems are used by Artificial Intelligence (AI) programs to perceive specific objects in class as subjects of interest. The systems sort objects in images into groups where objects with similar characteristics are placed together, while others neglected
### How does YOLOV3 work?
YOLO is a Convolutional Neural Network (CNN) for performing object detection in real- time. CNNS are classifier-based systems that can process input images as structured arrays identify patterns between them. YOLO has the advantage of being much faster than other networks and still maintains accuracy. 

![image](https://user-images.githubusercontent.com/29221802/134807996-41826598-3c62-4886-b960-2b7ffa5705fd.png)

It allows the model to look at the whole image at test time, so its predictions are convolutional network algorithms "score" regions based on their similarities to predefined classes. 
High-scoring regions are noted as positive detections of whatever class they most closely identify with. example, of traffic, YOLO can be used to detect different kinds of vehicles depending on which regions of the video score highly in comparison to predefined classes of vehicles.

![image](https://user-images.githubusercontent.com/29221802/134808079-1764c4e3-6a4d-478f-9202-2525cf05c076.png)

### Class Confidence and Box Confidence Scores 
Each bounding The confidence Score is the value of how probable a class is contained by that box, as well as how accurate that bounding box is. 
The bounding box width and height (w and h) is first set to the width and height of the image given. Then, x and y are offsets of the cell in question and all 4 bounding box between 0 and 1. Then, each cell has 20 conditional class probabilities implemented by the YOLOV3 algorithm. 
The class confidence score for each final boundary box used as a positive prediction is conditional class probability in this context is the probability that the detected object is part of certain (the class object of interest's identification), prediction, therefore, has 3 values of h, w, and depth. 
There is some math that then takes place involving spatial dimensions of the images and the tensors used in order to produce boundary box predictions, but that is complicated. If you are interested in learning what happens during this stage, I suggest the YOLOV3 Arxiv paper linked at the end of this article. 
For the final step, the boundary boxes with high confidence scores (more than 0.25) are kept as final predictions.

![image](https://user-images.githubusercontent.com/29221802/134808398-8d49488a-8334-4c67-afc8-7bce3e291996.png)

###  Load and use the pre-trained YOLO model. 
```terminal
MODEL = 'yolo\yolov3-face.cfg'
WEIGHT = 'yolo\yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

### Get the detection from YOLO model.
```terminal
IMG_WIDTH, IMG_HEIGHT = 416, 416
# Making blob object from original image
blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)

# Set model input
net.setInput(blob)

# Define the layers that we want to get the outputs from
output_layers = net.getUnconnectedOutLayersNames()

# Run 'prediction'
outs = net.forward(output_layers)
```

## Classification Model

#### __Model architecture__

#### __Results__
