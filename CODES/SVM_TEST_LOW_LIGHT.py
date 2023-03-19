# Import the libraries
import pickle
import os
import cv2
import numpy as np
import glob
from skimage.feature import hog

# Get the file names of all unlabelled images in the folder
folder_path = "C:/Users/ashut/Desktop/Course Project_1/Devnagri/MyLowIlluminatedDigits"
file_extension = '.jpg' # or use '*'
file_names = glob.glob(os.path.join(folder_path, '*' + file_extension))

# Load and resize each image
data = []
for file_name in file_names:
    img = cv2.imread(file_name)

    ## For low illuminated digits only. Use this Gaussian Blur
    #img = cv2.GaussianBlur(img, (13, 13), 0)
    ##

    ## For Histogram equalization Use this
    # Perform contrast stretching using histogram equalization
    # img = cv2.imread(file_name,0)
    # equ = cv2.equalizeHist(img)
    ##

    ##  For CLAHE method only
    # Create a CLAHE object with default parameters
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(2,2))

    # Apply CLAHE to the image
    # equ = clahe.apply(img)
    #img = cv2.resize(equ, (32, 32))
    ##

    # Resizing the image to 32X32
    img=cv2.resize(img,(32,32))

    # Converting to grayscale image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's threshold
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Normalizing the image
    image = np.array(image) / 255.0

    # Flattening the image 
    images_flat = image.reshape(image.shape[0], -1)
    hog_features = hog(images_flat, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    data.append(hog_features)
    
# Load the SVM model
filename = 'svm_model_1.sav'
loaded_model=pickle.load(open(filename, 'rb'))

# Initialize X_train and y_train
X_train = np.array([])
y_train = np.array([])

# Predict labels for each test image
for i in range(len(data)):
    prediction = loaded_model.predict([data[i]])[0]
    file_name = os.path.splitext(os.path.basename(file_names[i]))[0]
    print("File name: {}, Predicted label: {}".format(file_name, prediction))

 