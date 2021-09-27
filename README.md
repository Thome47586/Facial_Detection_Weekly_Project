# Facial Detection Project
__Date:__ 26th of September 2021

__Authors:__ 

 + __Tuan Le__: https://github.com/tuanle163
 + __Duy__: https://github.com/yuthefirst 
 + __Thome__: https://github.com/Thome47586 
 + __Vu Tran Huy__: https://github.com/vutrhuy81
 
# I. Overview
 The project is to detect our team members face and name. We use the OpenCV, YOLO and VGGFace model to train and detect our team member faces.
 + Total team members: 4

__Repo structure__
```
main.py

misc.ipynb

/dataset
    |
    |/dataset.zip

/Auto_Data_Collection_Code
    |
    |/Auto_Webcam_Capture.iypnb

/utils
    |
    |/Model_predict.py
    |/yolov3.py

/env
    |
    |/M1_facial_detection.yml

/model
```

# II. Requirement and Dataset

## __Requirements__
Following packages that you need to install:

+ python>=3.8
+ opencv-contrib-python
+ tensorflow
+ numpy
+ matplotlib

Then git clone our repo:

```terminal
git clone https://github.com/tuanle163/Facial_Detection_Weekly_Project
```

OR, You can create an Anaconda environment using our following __.yml__ files:

+ __Macbook M1 (ARM64 CPU):__ 

```terminal
conda env create create -f ./env/M1_facial_detection.yml
```

+ __Macbook Intel CPU: (Updating)__
```terminal
conda env create create -f ./env/MacIntel_facial_detection.yml
```
+ __Window Intel CPU: (Updating)__
```terminal
conda env create create -f ./env/WinIntel_facial_detection.yml
```

## __Dataset__

The dataset we used to train our model can be found here: https://github.com/tuanle163/Facial_Detection_Weekly_Project/tree/main/dataset

Or you can download it from Google Drive link: [Link](https://drive.google.com/file/d/1NYiarTKMgdsrWHXcRhmVblyPY3kJmuvj/view?usp=sharing)

The dataset was collected by capturing image from the webcam of each team members. The code to auto collect the image is in the following directories:

+ For using on Google Colab: [__Link__](https://github.com/tuanle163/Facial_Detection_Weekly_Project/tree/main/Auto_Data_Collection_Code)

The code will capture an image on webcam and using YOLOV3 model to finding the human face in the image. It is then crop out the face and save it in dataset folder. 

The dataset has 4 classes: 
+ __TuanLe:__ 600 images
+ __Thome:__ 611 images
+ __Duy:__ 600 images
+ __Vutrhuy81:__ 600 images

## __Model files__
YOLOv3 model file link:
+ Model: [Link](https://drive.google.com/file/d/1g9TarspT_H5wITwHYm5wLZLoofdsN3np/view?usp=sharing)
+ Weights: [Link](https://drive.google.com/file/d/1nFexKK2fRwWCB0vZmhpXML-h5FC_jCu7/view?usp=sharing)

VGGFace model file link:
+ Model: [__Link__](https://drive.google.com/file/d/1fnPAz_Mu-_at2BidQE_GdDGsIGxFrwOq/view?usp=sharing)

__Please, download the models and weight from the links above and save to <span style="color:Tomato;">*/model*</span> folder__


# III. Model Structure
![image](https://user-images.githubusercontent.com/29221802/134808537-bd1df0a8-5ebc-4f36-ad0b-13a1a2ebd38f.png)

## __YOLOV3 Model for human face detection__
![image](https://user-images.githubusercontent.com/29221802/134807498-77eb1ed1-58ac-4cba-bdce-d1be91367c1f.png)

### __What is YOLOV3?__

YOLOV3 (You Only Look Once, Version 3) is real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. YOLO uses features learned by a deep convolutional neural network to detect an object. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi. 

The first version of YOLO was created in 2016, and version 3, which was made two years later in 2018. YOLOV3 is an improved version of YOLO and YOLOV2. YOLO is implemented using the Keras or OpenCV deep learning libraries.

Object classification systems are used by Artificial Intelligence (AI) programs to perceive specific objects in class as subjects of interest. The systems sort objects in images into groups where objects with similar characteristics are placed together, while others neglected.

### __How does YOLOV3 work?__

YOLO is a Convolutional Neural Network (CNN) for performing object detection in real- time. CNNS are classifier-based systems that can process input images as structured arrays identify patterns between them. YOLO has the advantage of being much faster than other networks and still maintains accuracy. 

![image](https://user-images.githubusercontent.com/29221802/134807996-41826598-3c62-4886-b960-2b7ffa5705fd.png)

It allows the model to look at the whole image at test time, so its predictions are convolutional network algorithms "score" regions based on their similarities to predefined classes. 

High-scoring regions are noted as positive detections of whatever class they most closely identify with. example, of traffic, YOLO can be used to detect different kinds of vehicles depending on which regions of the video score highly in comparison to predefined classes of vehicles.

![image](https://user-images.githubusercontent.com/29221802/134808079-1764c4e3-6a4d-478f-9202-2525cf05c076.png)

### __Class Confidence and Box Confidence Scores__
Each bounding The confidence Score is the value of how probable a class is contained by that box, as well as how accurate that bounding box is. 

The bounding box width and height (w and h) is first set to the width and height of the image given. Then, x and y are offsets of the cell in question and all 4 bounding box between 0 and 1. Then, each cell has 20 conditional class probabilities implemented by the YOLOV3 algorithm. 

For the final step, the boundary boxes with high confidence scores (more than 0.25) are kept as final predictions.

![image](https://user-images.githubusercontent.com/29221802/134808398-8d49488a-8334-4c67-afc8-7bce3e291996.png)

###  Load and use the pre-trained YOLO model. 

```python
MODEL = 'yolo\yolov3-face.cfg'
WEIGHT = 'yolo\yolov3-wider_16000.weights'
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

### Get the detection from YOLO model.
```python
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

## __Classification Model__
### __Image Preprocessing__

### __Model architecture__

### __Results__
