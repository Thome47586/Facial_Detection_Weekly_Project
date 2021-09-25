import cv2
import os
import time

# Load YOLO 3
################################################################################################

MODEL = 'yolo/yolov3-face.cfg'
WEIGHT = 'yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

################################################################################################
IMG_WIDTH, IMG_HEIGHT = 416, 416

try:
    directory = input('Please enter your name:')
    parent_dir = 'faces'
    path = os.path.join(parent_dir, directory)
    os.makedirs(path)
    print(f'Create the folder name {directory} at {path}')
except FileExistsError:
    print('Folder\'s name Existed')

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

count = 0
while True:
    count = count + 1
    ret, frame = cap.read()

    # Making blob object from original image
    # [blobFromImage] creates 4-dimensional blob from image. 
    # Optionally resizes and crops image from center, subtract mean values, 
    # scales values by scalefactor, swap Blue and Red channels.
    blob = cv2.dnn.blobFromImage(frame, 
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)

    # Set model input
    # net.setInput() — to set the input data into the model
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    # net.getUnconnectedOutLayersNames() - returns all the output layers' name
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    # net.forward() - to pass data through the model 
    #(similar to fit and predict in sklearn)
    outs = net.forward(output_layers)

################################################################################################

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
            # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1] # take probability of each class
                # Extract position data of face area (only area with high confidence)
            if confidence > 0.5: # Chưa hiểu cách tính ở dưới
                center_x = int(detection[0] * frame_width) 
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                        
                # Find the top left point of the bounding box 
                # topleft_x = center_x - (height/2)
                # topleft_y = center_y - (width/2)
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
        left = int(box[0]-40)
        top = int(box[1]-30)
        width = int(box[2]+80)
        height = int(box[3]+50)

        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        cv2.rectangle(result, (left,top), (left+width, top+height), (0,255,0), 2)
        
        # Display text about confidence rate above each box
        text = f'Number of confident: {confidences[i]:.2f}'
        ### YOUR CODE HERE
        cv2.putText(result, text, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        crop_img = result[top: top + height, left : left + width]
        
        # frame is now the image capture by the webcam (one frame of the video)

############################################################################################################

        cv2.imshow('Input', result)
        time.sleep(1)
        # save file
        cv2.imwrite(os.path.join(path, f'{directory}_{count}.jpg'), crop_img)
    
    if count == 300:
        break
    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


