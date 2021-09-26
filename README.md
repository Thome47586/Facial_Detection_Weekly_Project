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
## Image Preprocessing


## YOLOV3 Model
![image](https://user-images.githubusercontent.com/29221802/134807498-77eb1ed1-58ac-4cba-bdce-d1be91367c1f.png)


## Classification Model

#### __Model architecture__

#### __Results__
