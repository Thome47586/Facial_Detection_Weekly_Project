import cv2
import os
import time
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array

# Load YOLO 3
################################################################################################
MODEL = 'Facial_Detection/yolo/yolov3-face.cfg'
WEIGHT = 'Facial_Detection/yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416
################################################################################################

# Load model predict
model = tf.keras.models.load_model('Facial_Detection/Model/Tuan_model.h5')
labels = ['Duy', 'Thome', 'TuanLe', 'Vutrhuy81']

# function predict
def predict_face(frame):

    img        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img        = tf.image.resize(img, (224,224))
    img        = img/255
    img_array  = image.img_to_array(img)
    img_array  = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    make_label = np.argmax(prediction)
    define_class = labels[make_label]
    
    return define_class
################################################################################################

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # net.forward() - to pass data through the model, like .fit on keras
    outs = net.forward(output_layers)
    ################################################################################################
    # Identify width and height of frame
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
            # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1] # take probability of facial class
                # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width) 
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                        
                # Find the top left point of the bounding box 
                topleft_x = center_x - (width/2)
                topleft_y = center_y - (height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    ################################################################################################

    final_boxes = []
    result = frame.copy()
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])

        # Draw bouding box with the above measurements
        cv2.rectangle(result, (left,top), (left+width, top+height), (0,255,0), 2)

        # crop face
        crop_img = result[top: top + height, left : left + width]

        # Call function predict
        name_label = predict_face(crop_img)
        cv2.putText(result, name_label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        ############################################################################################################

        cv2.imshow('Input', result)
    c = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()


