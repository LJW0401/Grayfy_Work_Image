import tkinter as tk
import tkinter.ttk
import tkinter.messagebox
import os
import os.path
import cv2
import numpy as np
from PIL import Image, ImageTk

'''读取图像'''
# Read the image as grayscale
img = cv2.imread('E:\homework_img_deal\original/w12-1.jpg')

'''算法实现部分'''
# 将图像转换为HSV颜色空间
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义黑色的HSV范围
lower_black = (0, 0, 0)
upper_black = (360, 100, 150)

# 创建掩膜
mask = cv2.inRange(hsv_image, lower_black, upper_black)
mask_ = 255-mask
# 对图像进行分割，提取黑色部分
result = cv2.bitwise_and(img, img, mask=mask)

'''算法实现部分'''
# Apply adaptive thresholding
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img_filtered=thresh.copy()

for i in range(7):
    # Apply median filter to remove salt-and-pepper noise
    img_filtered = cv2.medianBlur(img_filtered, 3)
    # Apply bilateral filter to preserve edges and details
    img_filtered = cv2.bilateralFilter(img_filtered, 7, 75, 75)

'''图像合成'''
# result = 255-(mask & (255-img_filtered))
# 设置合成比例（两张图像的权重）
alpha = 0.3
beta = 0.7
blended_image = cv2.addWeighted(mask_, alpha, img_filtered, beta, 0.0)

'''展示图像'''
# Display the thresholded image
# cv2.imshow('Mask Image', cv2.resize(mask_,(0,0),fx=0.25,fy=0.25))
# cv2.imshow('Result Image', cv2.resize(result,(0,0),fx=0.25,fy=0.25))
cv2.imshow('Blended Image', cv2.resize(blended_image,(0,0),fx=0.2,fy=0.2))
# cv2.imshow('Image', cv2.resize(np.hstack((mask_,img_filtered,result)),(0,0),fx=0.2,fy=0.2))
cv2.waitKey(0)
cv2.destroyAllWindows()

