import tkinter as tk
import tkinter.ttk
import tkinter.messagebox
import os
import os.path
import cv2
import numpy as np
from PIL import Image, ImageTk

# Read the image as grayscale
img = cv2.imread('E:\homework_img_deal\original/w1-1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img_filtered=thresh.copy()

for i in range(7):
    # Apply median filter to remove salt-and-pepper noise
    img_filtered = cv2.medianBlur(img_filtered, 3)
    # Apply bilateral filter to preserve edges and details
    img_filtered = cv2.bilateralFilter(img_filtered, 7, 75, 75)



# Display the thresholded image
cv2.imshow('Thresholded Image', cv2.resize(np.hstack((thresh,img_filtered)),(0,0),fx=0.25,fy=0.25))
cv2.waitKey(0)
cv2.destroyAllWindows()

