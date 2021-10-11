# -*- coding: utf-8 -*-
"""
@author: shubhangi
"""

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


'''
import cv2
video_capture = cv2.VideoCapture(0)

cv2.namedWindow("Window")

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Window", frame)

    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
'''

face_cascade = cv2.CascadeClassifier(r'A:\PROJ fol\code_rel\haarcascade_frontalface_alt2.xml')

#prototxtPath = r"A:\PROJ fol\code_rel\deploy.prototxt"
#weightsPath = r"A:\PROJ fol\code_rel\res10_300x300_ssd_iter_140000.caffemodel"
#face_cascade = cv2.dnn.readNet(prototxtPath, weightsPath)
classifier =load_model(r'A:\PROJ fol\code_rel\mask\Emotion_model.h5')



class_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

cap = cv2.VideoCapture(0)
#cv2.namedWindow("Window")
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,
               cv2.WINDOW_FULLSCREEN)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Window", gray)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            if label=='angry':
                cv2.putText(frame,'Anger is your biggest enemy',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            if label=='neutral':
                cv2.putText(frame,'Be HAPPY and smile:)',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            if label=='happy':
                cv2.putText(frame,'Happiness is the state of activity',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            if label=='surprise':
                cv2.putText(frame,'Let life surprise you',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            if label=='sad':
                cv2.putText(frame,'Be Happy',(20,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow('Emotion Detector',frame)
    #cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()