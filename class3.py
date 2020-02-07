import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = 'D:/Downloads/indian_coins.jpg'
img = cv2.imread(img_path, 0)

# Noise Reduction/Median Filter
kernel = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(img, -1, kernel)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blurred), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Thresholding
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)
cv2.waitKey(0)
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)

# Adaptive Thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
cv2.imshow("Mean Thresh", thresh)
cv2.waitKey(0)

# Laplacian Filter for Edge Detection
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
cv2.waitKey(0)

# Sobel Filter for edge detection
# When only in one axis, Laplacian is overkill
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)

# Blurring + Canny for better edge detection
image = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Blurred", image)
canny = cv2.Canny(image, 30, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)


# Auto canny function to decide okayish T1 T2 for canny
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


cv2.imshow("Auto Canny", auto_canny(img, 0.75))
cv2.imshow("Canny", canny)
cv2.waitKey(0)

# image = cv2.medianBlur(image,3)
image = cv2.GaussianBlur(img, (7, 7), 0)
# image_eq = cv2.equalizeHist(image)
cv2.imshow("Blurred", image)
cv2.waitKey(0)
canny = auto_canny(image, 0.5)
# canny = cv2.Canny(image, 10, 200)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
coins = img.copy()
cv2.drawContours(coins, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)

