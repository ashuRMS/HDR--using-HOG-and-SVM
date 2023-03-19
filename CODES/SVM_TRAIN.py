# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# Load the data of all languages
train_dir = "C:/Users/ashut/Desktop/Course Project_1/Overall/Train_data"
test_dir = "C:/Users/ashut/Desktop/Course Project_1/Overall/Test_data"
directory="C:/Users/ashut/Desktop/Course Project_1/Overall"

# Test and train images
def load_images(directory):
    images = []
    labels = []
    for folder_name in os.listdir(directory):
        label = str(folder_name)
        folder_path = os.path.join(directory, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # Changing colorscale to grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            # Applying Otsu's threshold
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            images.append(image)
            labels.append(label)
    return images, labels

train_images,train_labels=load_images(train_dir)
test_images,test_labels=load_images(test_dir)

## Pre-processing images
# The images are of shape 32X32

#Normalizing the images
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0

# Flattening the images to numpy 1d array
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

#Extracting hog features for train and for test images
from skimage.feature import hog
features_train = []
for image in train_images:
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features_train.append(hog_features)

features_test = []
for image in test_images:
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features_test.append(hog_features)

# Training using Linear SVM classifier
from sklearn import svm

#Load the linear SVM
svm_model_1 = svm.SVC(kernel='linear', C=1, gamma='auto')
svm_model_1.fit(features_train, train_labels)

# Save the trained model to a file
import pickle
with open('svm_model_1.sav', 'wb') as f:
    pickle.dump(svm_model_1, f)

# Predicting the labels
predicted_labels = svm_model_1.predict(features_test)

#Performnace Metrics for SVM model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
import pandas as pd
import seaborn as sns
conf_matrix=confusion_matrix(y_true=test_labels,y_pred=predicted_labels)

class_names=['Digit_0B','Digit_0D','Digit_0T','Digit_1B','Digit_1D','Digit_1T','Digit_2B','Digit_2D','Digit_2T','Digit_3B','Digit_3D','Digit_3T',
             'Digit_4B','Digit_4D','Digit_4T','Digit_5B','Digit_5D','Digit_5T','Digit_6B','Digit_6D','Digit_6T',
             'Digit_7B','Digit_7D','Digit_7T','Digit_8B','Digit_8D','Digit_8T','Digit_9B','Digit_9D','Digit_9T']

sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# assume that 'predicted_labels' and 'ground_truth_labels' are numpy arrays containing predicted and ground truth labels, respectively
ground_truth_labels=test_labels
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
precision = precision_score(ground_truth_labels, predicted_labels, average='macro')
recall = recall_score(ground_truth_labels, predicted_labels, average='macro')
f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')

# Display classification report
per_class_accuracy = classification_report(y_true=ground_truth_labels, y_pred=predicted_labels)

# create a pandas dataframe to display the performance metrics
data = {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1-score': [f1]}
df = pd.DataFrame(data)
print(df)

