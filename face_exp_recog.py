#Importing the necessary libraries
import cv2
import numpy as np
import keras.utils as image
import warnings
from keras.models import  load_model
import numpy as np

# load model
model = load_model('models/vgg16.h5')

#Loading the haarcascade_frontalface_default.xml file for finding the faces
face_haar_cascade = cv2.CascadeClassifier('xml_files/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    #Captures frame and returns boolean value and captured image
    if not ret:
        continue

    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) 
    faces_detected = face_haar_cascade.detectMultiScale(img, 1.2, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness = 1) # Displays the rectangle on the face

        # Image Processing
        roi = img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from image
        roi = cv2.resize(roi, (48, 48))
        img_pixels = image.img_to_array(roi)
        img_pixels /= 255
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels) #Emotion Prediction

        # find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise') # Different Classes in the Model
        predicted_emotion = emotions[max_index]
        confidence_val = int(max(max(predictions)*100)) # % of Confidence
        emotion_text = str(predicted_emotion) + "%: " + str(confidence_val) + "%"

        # The below colors are used to display different emotions
        if predicted_emotion == str('angry'):   
            cv2.putText(test_img, emotion_text, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif predicted_emotion == str('fear'):
            cv2.putText(test_img, emotion_text, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif predicted_emotion == str('happy'):
            cv2.putText(test_img, emotion_text, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif predicted_emotion == str('neutral'):
            cv2.putText(test_img, emotion_text, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128 , 0, 128), 1)
        elif predicted_emotion == str('sad'):
            cv2.putText(test_img, emotion_text, (525, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 234, 0), 1)
        elif predicted_emotion == str('surprise'):
            cv2.putText(test_img, emotion_text, (510, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    resized_img = cv2.resize(test_img, (800, 500))
    cv2.imshow('Facial Expression Recognition', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows