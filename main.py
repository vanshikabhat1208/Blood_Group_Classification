import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC

# Load image and convert to grayscale
img = cv2.imread('blood_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding to segment blood cells
ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY) 

# Morphological opening to remove noise 
kernel = np.ones((3,3), np.uint8) 
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Segment and extract individual cells
cells = cv2.connectedComponentsWithStats(opening)  

# Extract features like area, perimeter, shape, texture etc. 
features = []
for cell in cells[3]:
    area = cell[4]
    perimeter = cell[3]
    circularity = 4*np.pi*area/perimeter**2
    glcm = graycomatrix(cell, [5], [0])
    contrast = graycoprops(glcm, 'contrast')[0,0]
    features.append([area, perimeter, circularity, contrast])

# Train SVM model on training data   
X_train = training_features
y_train = training_labels
clf = SVC()
clf.fit(X_train, y_train)

# Predict on test data
X_test = test_features
y_pred = clf.predict(X_test) 

# Print accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))