import numpy as np
import cv2
from matplotlib import pyplot as plt

filepath = 'E:\\GAYANI\\character_recognition\\id_images\\DSC.jpg'

img = cv2.imread(filepath)
gray_img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
gray_img = cv2.resize(gray_img, (255,255))

img = cv.imread('sudoku.png',0)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

c = croped[38*4:5*38,:]
cv2.imshow('my', c)
