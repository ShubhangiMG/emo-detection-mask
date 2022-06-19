# Emotion detection from facial expressions (extended for images with facemask)

This project is an extension to the standard emotion detection problem using the facial expressions. It extends the usage for the COVID-19 scenario where the facial expressions are hugely obstructed by the usage of face masks. 
The classification model is built using the Convolutional Neural Networks for 7 classes of emotions-.
![image](https://user-images.githubusercontent.com/50448485/163726982-889b371a-aa14-4aa4-bdd5-902334c8e129.png)

Emotion_model.h5 file contains the model trained on the images in which face are not covered with masks. The dataset has been taken from Kaggle- FER 2013 dataset.
The model obtained an accuracy of 63.62%.
In vidTRY.py, the classificatoin model is implemented for real time input from the web camera. For detection of face from the web cam input, a pretrained face detection model from OpenCV has been taken, named as HaarCascade model (haarcascade_frontalface_alt2.xml in the repo)


The complete flow of real time implementation is shown-
![image](https://user-images.githubusercontent.com/50448485/163726923-865dca7e-d7df-4306-964c-d66c04e769c2.png)

The model is then extended for artificially genetated dataset that consists of images with masked faces for 7 classes of emotions.
Same process has been performed for masked dataset but with improved CNN structure to achieve better accuracy than that of model for unmasked dataset. 
Dataset created process- reference. Explain process(technically) (masking.py inside mask folder)
The artificial dataset is created taking images of the FER2013 dataset and putting mask over all of the images. 
Here is how the process works for a single image-
1. The image is looked for landmarks such as chin points, nose bridge and width.
2. The dimensions of the mask image are adjusted using to the landmark positions.
3. The mask is put on the image to create the final masked image.
Emotion_model.h5 in mask folder has mask model %accuracy%

Same real time implementation
