import cv2

##############################################################################################################
MODEL = 'yolo_weight/yolov3-face.cfg'
WEIGHT = 'yolo_weight/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

IMG_WIDTH, IMG_HEIGHT = 416, 416

def take_faces(frame):
##############################################################################################################

  # Making blob object from original image
  blob = cv2.dnn.blobFromImage(frame, 
                1/255, (IMG_WIDTH, IMG_HEIGHT),
                              [0, 0, 0], 1, crop=False)

  # Set model input
  net.setInput(blob)

  # Define the layers that we want to get the outputs from
  output_layers = net.getUnconnectedOutLayersNames()

  # Run 'prediction'
  outs = net.forward(output_layers)

  ##############################################################################################################

  blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
  # print(blobb.shape)

  ##############################################################################################################

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
          confidence = detection[-1]
          # Extract position data of face area (only area with high confidence)
          if confidence > 0.5:
              center_x = int(detection[0] * frame_width)
              center_y = int(detection[1] * frame_height)
              width = int(detection[2] * frame_width)
              height = int(detection[3] * frame_height)
              
              # Find the top left point of the bounding box 
              topleft_x = center_x - width/2
              topleft_y = center_y - height/2
              confidences.append(float(confidence))
              boxes.append([topleft_x, topleft_y, width, height])

  # Perform non-maximum suppression to eliminate 
  # redundant overlapping boxes with lower confidences.
  indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  ##############################################################################################################
  final_boxes = []
  result = frame.copy()
  for i in indices:
      i = i[0]
      box = boxes[i]
      final_boxes.append(box)
      # Extract position data
      x = int(box[0])
      y = int(box[1])
      w = int(box[2])
      h = int(box[3])
      # Draw bouding box with the above measurements
      cv2.rectangle(result, (x, y), (w+x, h+y), (0, 255, 0), 2)
    # # Display text about confidence rate above each box
      text = f'{confidences[i]:.2f}'
      cv2.putText(result, f'confident: {text}', (int(x), int(y-5)), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255) ,1)
      number_of_faces = len(indices)
    # Display text about number of detected faces on topleft corner
      cv2.putText(result, f'Number of face detected: {number_of_faces}', 
                  (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255) ,1)
    # frame is now the image capture by the webcam (one frame of the video)
  return x, y, w, h, result


