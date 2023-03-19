import pickle
import os
import cv2
import numpy as np
import glob
from skimage.feature import hog


# Get the file names of all unlabelled images in the folder
folder_path = "C:/Users/ashut/Desktop/Course Project_1/Devnagri/MyThinLineData"
file_extension = '.jpg' # or use '*'
file_names = glob.glob(os.path.join(folder_path, '*' + file_extension))

# Load and resize each image
data = []
for file_name in file_names:
    # Use this for converting to binary images
    img=cv2.imread(file_name,0)

    # Apply thresholding to convert the image to binary
    thresh_value = 128# Threshold value
    max_value = 255 # Maximum value to use with THRESH_BINARY

    # Applying binary threshold
    thresh, binary_img = cv2.threshold(img, thresh_value, max_value, cv2.THRESH_BINARY)

    # Dilating the images to increase the digit's thickness
    D_r=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    d_r=cv2.erode(binary_img,D_r) # Since image is black text on white background
    
    
    # Resizing the image
    image = cv2.resize(d_r, (32, 32))

    # Appling Otsu's threshold on the image
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Normalizing the image
    image = np.array(image) / 255.0

    # Flattening the 2d image to 1d image
    images_flat = image.reshape(image.shape[0], -1)

    # Extracting HOG features
    hog_features = hog(images_flat, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    data.append(hog_features)
    
# Load the saved SVM Model
filename = 'svm_model_1.sav'
loaded_model=pickle.load(open(filename, 'rb'))

# Initialize X_train and y_train
X_train = np.array([])
y_train = np.array([])

# Predict the labels
for i in range(len(data)):
    prediction = loaded_model.predict([data[i]])[0]
    file_name = os.path.splitext(os.path.basename(file_names[i]))[0]
    print("File name: {}, Predicted label: {}".format(file_name, prediction))
