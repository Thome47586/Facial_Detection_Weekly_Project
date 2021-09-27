import cv2
import tensorflow as tf
from utils import yolov3
from utils import Model_predict

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        # Call Yolo V3 function
        final_box = yolov3.yolo_v3(frame)

        # identify coordinate point of best box
        left, top, width, height = 0,0,0,0
        for i in range(len(final_box)):
            left = int(final_box[0])
            top = int(final_box[1])
            width = int(final_box[2])
            height = int(final_box[3])
        
        result = frame.copy()
        # Rectangle
        cv2.rectangle(result, (left,top), (left+width, top+height), (0,255,0), 2)
        crop_img = result[top: top + height, left : left + width]
        
        # Call function predict
        name_label = Model_predict.predict_face(crop_img)
        cv2.putText(result, name_label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        cv2.imshow('Input', result)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()