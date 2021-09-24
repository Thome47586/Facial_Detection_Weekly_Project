import cv2
import face_dectection
import os

##############################################################################################################
try:
  directory = input('Please enter your name: ')
  parent_dir = "faces"
  path = os.path.join(parent_dir, directory)
  print(path)
  os.mkdir(path)
  print(f"Create the folder name {directory} at {path}")
except FileExistsError:
  print('Folder\'s name Existed ')
##############################################################################################################
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")
count = 0
while True:
  count = count + 1
  ret, frame = cap.read()
  x, y, w, h, result = face_dectection.take_faces(frame)
  crop_img = result[y:y+h, x:x+w]
  
  # frame is now the image capture by the webcam (one frame of the video)
  cv2.imshow('Input', result)
  cv2.imwrite(os.path.join(path , f'{directory}_{count}.jpg'), crop_img)

  c = cv2.waitKey(1)  
  # Break when pressing ESC
  if c == 27:
      break

cap.release()
cv2.destroyAllWindows()


