# Facical_Detection_Weekly_Project
 
Display facial detection with 3 tab on VS code:

1. Main:
		a. Objective:
			- In charge inputs and outputs
			- Coordinate Yolo v3 and predict_face to organize
		b. Workflow
			- Main: 			Use library OpenCV open webcam to take frames
			- Yolo V3: 			Call and push frames into tab Yolo V3 take out a best bounding box
			- Main: 			Identify coordinates point of the box
			- Main: 		  	Determine rectangle of face to display and crop this face to predict 
			- Model_ predict: 	Transfer crop face into tab Model_predict to define who are they and take the name to display
2. Yolo_V3
		a. Objective:
			- Define a best bounding box and split coordinate points
		b. Workflow
			- Call model and weights specializing in facial detection
			- Train model
			- Define bounding boxes have confident more than 0.5 and eliminate redundant overlapping boxes
			- Take best bounding box pass out Main tab

3. Model_predict
		a. Objective:
			- Predict who are they, result is a name
		b. Workflow
			- Load model and labels
			- Take crop face into function and resize, preprocess 
			- Let's model predict the face
			- Push out result to Main tab
4. Enviroment
		- Python3
		- Tensorflow 2.6
		- OpenCV 
		- Numpy
		