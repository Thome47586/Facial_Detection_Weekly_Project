import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model VGG16 face to predict
model = tf.keras.models.load_model('model/Duy_face_recognition.h5')
labels = ['Duy', 'Thome', 'TuanLe', 'Vutrhuy81']

# function predict
def predict_face(face):
    try:
        img        = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img        = tf.image.resize(img, (224,224))
        img_array  = image.img_to_array(img)
        img_array  = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        make_label = np.argmax(prediction)
        define_class = labels[make_label]
        return define_class
    except:
        print('Wait for a few seconds')