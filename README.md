# HDR--using-HOG-and-SVM
This repository contains my Course project in Computer Vision. The paper is titled "Robust handwritten digit recognition using SVM and HOG."
## Requirements.
- Python 3.10.7
- OpenCV
- Pandas
- Numpy
- matplotlib
- skiit-learn
- pickle 
- seaborn
## Project Details
A machine learning model SVM is used to learn the multi - digit classification. The digits were taken from 3 languages named Devanagari, Bangla, and Telugu. The aim is to recognize the digits as well as the language of the digit under various conditions.
- Description of approach: Entire data set of images were preprocessed in which we resized the images, converted them to grayscale, and extracted HOG features. HOG features is a very efficient way of describing hand written text which was the input for the classifier. Performance metrics such as accuracy, precision, recall and F-1 scores are used for gauging the results from the classifier.
- About the data sets used: We have used three languages for the task. Devanagri (Hindi), Bangla and Telugu. Data-sets are taken from Kaggle.com and CMATTERdb project.
The Devanagari dataset has 17,000 images for training and 3000 images for testing. The Bangla dataset has 4000 images for training and 2000 images for testing. The Telugu dataset has 2000 images for training and 1000 for testing.
The Digit-labels are created in this way -->; for example, Digit_6T represents the digit '6' in Telugu. Similarly, the letter 'B' for Bangla and 'D' for Devanagari is used. Links for the same can be found below.
 1. [Devanagri Data-set](https://www.kaggle.com/datasets/anurags397/hindi-mnist-data)
 2. [Bangla Data-set](https://code.google.com/archive/p/cmaterdb/)
 3. [Telugu Data-set](https://www.kaggle.com/datasets/anurags397/hindi-mnist-data)
- Training the classifier: Training was done with the help of a linear SVM classifier using the fact that HOG features are linearly separable. We have trained single language data set and also a mixed language dataset, which was then used for testing. The output of test is the numeral and its language. We also learned from our literature survey that CNN or K-CNN can be used for improved performance, which is out of this project's scope.
## Accuracy using SVM classifier.
1. For Devanagri- 97.9%
2. For Bangla- 95.1%
3. For Telugu- 96.7%
4. Mixed model- 93.9%
## Experimenting with new evaluation set.
- On Hand crafted images: Custom unlabelled data set of handwritten digits was created to test the model. Accuracy was found to be 80%.
- On digits written on crumbled paper: The digits were written on a crumbled white paper. This was dome to make it harder for the model to classify the digits.       Accuracy-70%.
- Affine transformed digits: Randomly rotated digits were given as input to the model-Accuracy- 20%. Poor performance was expected.
- On Lined-Digit Data-set: Few random lines were drawn on the digit image and given as input to the model. Accuracy-70%.
- On varying digit thickness: Writing pressure can vary from person to person, so varying thickness digit images were also tested. But initial results show poor         accuracy of 20%. But after using cv2.erode function, which dilates the thickness of the digit digitally. Since, the images were black text on whitebackground, we have used erode function  we got an accuracy of 70%.
- Low illuminated digits: Classifiers or any other vision learning algorithms have a hard time recognizing the images taken in low light. Although computational         photography has improved this condition a lot still, there are challenges in low-light images. Model without any preprocessing, in our case, gave an accuracy of 50%   only. But after a few tweaks, we were able to get it to 60%. Including removing blur using the gaussian blur technique, which improved the accuracy to 60%.
## Limitations of the approach.
1. This model is susceptible to affine transformed images. The model can predict the upright images with reasonable accuracy, but the model performs poorly when we introduce rotation in the image. One solution to overcome this limitation would be to use higher-level ML, such as CNN.
2. The low-light performance of SVM is quite poor. Even better preprocessing ideas such as Gaussian blur, Histogram equalization, and CLAHE methods failed to improve digit recognition performance. Thus more research and effort are needed in challenging lighting conditions.
## Conclusions.
The model was able to predict labels successfully with the datasets available online. But the same model showed variations in accuracy from good to poor in the custom data sets where we have used crumbled sheets, affine images, lined-dataset images, etc. The model struggled to predict labels for affine-transformed images and has poor low-light performance. There is further scope of work where we can implement the CNN algorithm to improve performance on low-light and affine images. CNN is computationally intensive, but it requires less preprocessing than other classifiers, such as SVM. Since SVM is a linear classifier, there are tasks that it cannot handle, which are overcome by using CNN. Also, neural networks ease of use is one of the advantages over SVM.
