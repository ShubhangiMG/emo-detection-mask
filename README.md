# Emotion detection from facial expressions (extended for images with facemask)

This project is an extension to the standard emotion detection problem using the facial expressions. It extends the usage for the COVID-19 scenario where the facial expressions are hugely obstructed by the usage of face masks. 
The classification model is built using the Convolutional Neural Networks for 7 classes of emotions-.

Emotion_model.h5 file contains the model trained on the images in which face are not covered with masks. The dataset has been taken from Kaggle- FER 2013 dataset.
%mention accuracy%
In vidTRY.py, the classificatoin model is implemented for real time input from the web camera. For detection of face from the web cam input, a pretrained face detection model from OpenCV has been taken, named as HaarCascade model (haarcascade_frontalface_alt2.xml in the repo)


Same process has been performed for masked but with improves CNN structure to achieve better accuracy than the accuracy of previous model for masked dataset. 
Dataset created process- reference. Explain process(technically) (masking.py inside mask folder)

Emotion_model.h5 in mask folder has mask model %accuracy%

Same real time implementation
