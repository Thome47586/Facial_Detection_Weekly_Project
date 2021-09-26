import cv2

# Load YOLO 3
MODEL = 'Facial_Detection/yolo/yolov3-face.cfg'
WEIGHT = 'Facial_Detection/yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH, IMG_HEIGHT = 416, 416


def yolo_v3(frame):
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

    # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Determine best bounding box
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.extend(box)

    return final_boxes