# 7

import cv2

img_path = 'D:\Downloads\image1.jpg'
image = cv2.imread(img_path)
img = image
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("Original", img)
cv2.waitKey(0)

# 8
import cv2

img_path = 'D:\Downloads\image2.jpeg'
image = cv2.imread(img_path)
img = image
cv2.imshow("Original", img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = (img[:, :, 0], img[:, :, 1], img[:, :, 2])
eq = cv2.equalizeHist(V)
cv2.imshow("Histogram Equalization", eq)
cv2.waitKey(0)

import numpy as np

(B, G, R) = cv2.split(img)  # = (img[:, :, 0], img[:, :, 1], img[:, :, 2])
G = np.zeros(img.shape[:2], dtype='uint8')
merged = cv2.merge([B, G, R])
cv2.imshow("merge", merged)
cv2.waitKey(0)
