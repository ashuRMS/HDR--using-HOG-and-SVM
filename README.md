# HDR--using-HOG-and-SVM
This repository contains my Course project in Computer Vision. The paper is titled "Robust handwritten digit recognition using SVM and HOG"
## Project Details
A machine learning model SVM is used to learn the digit classification. The digits are taken from 3 languages named Devanagari, Bangla and Telugu. The aim is to recognize the digits as well as the language of the numeral, under various conditions.
- Description of approach: Entire data-set of images were preprocessed in which we resized the images, convetred them to grayscale and extracted HOG features. HOG features is very efficient way of describing hand written text which was the input for the classifier. Performance metrics such as accuracy, precision, recall and F-1 scores are used for gauging the results from the classifier.
- About the data-sets used: We have used 3 languages for the task. Devanagri (Hindi), Bangla and Telugu. Data-sets are taken from Kaggle.com and CMATTERdb project. Digit-images are renamed, for example Digit_6T represent digit '6' in Telugu language. Similarly B for Bangla and D for Devanagari is employed. links for the same can be found below.
 1. [Devanagri Data-set](https://www.kaggle.com/datasets/anurags397/hindi-mnist-data)
 2. [Bangla Data-set](https://code.google.com/archive/p/cmaterdb/)
 3. [Telugu Data-set](https://www.kaggle.com/datasets/anurags397/hindi-mnist-data)
- Training the classifier: Training was done with the help of SVM classifier using the fact that HOG features are linearly saperable. We have trained single langauge data set and also mixed language model which was then used for testing and getting output as name of the language and the digit recognized. We also learned that the CNN or K-CNN can be used for imporved performance, which is out of the scope of this project.
## Accuracy using SVM classifier.
1. For Devanagri- 97.9%
2. For Bangla- 95.1%
3. For Telugu- 96.7%
4. Mixed model- 93.9%
## Experimenting with new evaluation set.
- On Hand crafted images: Custom unlabeled data-set was created to test the model. Accuracy was found to be 80%.
- On digits written on crumbled paper: Paper was slightly crumbled which makes it harder for getting the results from the classifier. Accuracy-70%.
- Affine transformed digits: Randomly rotated few imgaes was given as input to model. Accuracy:20%.
- 
