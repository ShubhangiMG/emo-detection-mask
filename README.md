# Emotion detection from facial expressions (extended for images with facemask)

This project is an extension to the standard emotion detection problem using the facial expressions. It extends the usage for the COVID-19 scenario where the facial expressions are hugely obstructed by the usage of face masks. 
The classification model is built using the Convolutional Neural Networks (CNN) for 7 classes of emotions- angry, fear, happy, sad, surprise, disgust, neutral.

![image](https://user-images.githubusercontent.com/50448485/163726982-889b371a-aa14-4aa4-bdd5-902334c8e129.png)

Complete flow of approach used.

Emotion_model.h5 file contains the model trained on the dataset where faces in the images are not covered with masks. FER 2013 dataset is used for training and testing the model. The model obtained an accuracy of 63.62%. In vidTRY.py, the classificatoin model is implemented for real time input from the web camera. For detection of face from the web cam input, a pretrained face detection model from OpenCV has been taken, named as HaarCascade model (haarcascade_frontalface_alt2.xml in the repo).

![image](https://user-images.githubusercontent.com/50448485/163726923-865dca7e-d7df-4306-964c-d66c04e769c2.png)

Flow of real time implementation.


![image](https://user-images.githubusercontent.com/50448485/174656127-d0bc2451-e395-4849-b96e-1aafb2b708e6.png)

Real time implementation for input without mask.

The model is then extended for artificially genetated dataset that consists of images with masked faces for 5 classes of emotions-angry, fear, happy, sad, surprise.
Same process has been performed for masked dataset but with improved CNN structure to achieve considerable accuracy.
The artificial dataset is created taking images of the FER2013 dataset and putting mask over all of the images. 
Here is how the process works for a single image-
1. The image is looked for landmarks such as chin points, nose bridge and width.
2. The dimensions of the mask image are adjusted using to the landmark positions.
3. The mask is put on the image to create the final masked image.

The model Emotion_model.h5 created for masked dataset is present in the mask folder. It was also implemented for real time emotion detection when the person in front of the webcam is wearing a mask. The model was able to predict for 5 classes of emotions with an accuracy of 55.69%.

![image](https://user-images.githubusercontent.com/50448485/174656000-f66e9232-e10e-449f-bef4-1d65c64ec7ac.png)

Real time implementation for input with mask.
