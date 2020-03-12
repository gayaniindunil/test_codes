from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:\\GAYANI\\character_recognition\\id_images\\Nic_images\\DSC_0027.JPG' ,0)
image = cv2.resize(image,(800,800))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist,bins = np.histogram(image.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(image.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[image]
cv2.imshow('x',img2)
cv2.waitKey()

image = cv2.imread('E:\\GAYANI\\character_recognition\\id_images\\Nic_images\\DSC_0027.JPG' ,0)
image = cv2.resize(image,(800,800))
equ = cv2.equalizeHist(image)
res = np.hstack((image,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)

cv2.imshow('y',equ)
cv2.waitKey()


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)
cv2.imshow('clahe_2.jpg',cl1)
cv2.waitKey()