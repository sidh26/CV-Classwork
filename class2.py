import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = 'D:\Jatayu Unmanned Systems\dataset2\Elon.jpg'
image = cv2.imread(img_path)
img = image

print("max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))
print("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))  # x1+x2-256
print("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))  # x1-x2+256

M = np.ones(img.shape, dtype="uint8") * 180
added = cv2.add(img, M)
cv2.imshow("Added", added)  # gets brighter
cv2.waitKey(0)
# subtract makes darker

# Bitwise operations
rectangle = np.zeros(img.shape, dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)

circle = np.zeros(img.shape, dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)

m = cv2.bitwise_and(circle, rectangle)
cv2.imshow("M", m)
cv2.waitKey(0)

# Masking
mask = np.zeros(img.shape[:2], dtype="uint8")
(cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)

# grayscale Histogram
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", img)
cv2.waitKey(0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
# Very few white pixels
# Lotta black and less lotta but still lotta light grey

# Histogram equalization
img = cv2.imread(img_path)
img = cv2.resize(img, (500, 300))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
eq = cv2.equalizeHist(img)
eq_hist = cv2.calcHist([eq], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
plt.figure()
plt.title("Equalized Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(eq_hist)
plt.xlim([0, 256])
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([img, eq]))
cv2.waitKey(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
cv2.imshow("Histogram Equalization", np.hstack([img, cl1]))
cv2.waitKey(0)

# Blurring
# many image processing and computer vision functions,
# such as thresholding and edge detection,
# perform better if the image is first smoothed or blurred.

# Averaging blur
# The larger our kernel becomes, the more blurred our image will appear.
# (k,k) is convolution kernel or kernel
blurred_avg = np.hstack([
    cv2.blur(img, (3, 3)),
    cv2.blur(img, (5, 5)),
    cv2.blur(img, (7, 7))])
cv2.imshow("Averaged", blurred_avg)
cv2.waitKey(0)

# Gaussian Blur
# Gaussian blurring is similar to average blurring,
# but instead of using a simple mean, we are now using a weighted mean,
# where neighborhood pixels that are closer to the central pixel contribute more “weight” to the average.
blurred = np.hstack([
    cv2.GaussianBlur(img, (3, 3), 0),
    cv2.GaussianBlur(img, (5, 5), 0),
    cv2.GaussianBlur(img, (7, 7), 0)])
# The last parameter is our σ, the standard deviation in the x-axis direction. By setting this value to 0, we are instructing OpenCV to automatically compute them based on our kernel size.
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)

# Median Blur
# Traditionally, the median blur method has been most effective when removing salt-and-pepper noise,
# because each central pixel is always replaced with a pixel intensity that exists in the image.

blurred_avg = np.hstack([
    cv2.medianBlur(img, 3),
    cv2.medianBlur(img, 5),
    cv2.medianBlur(img, 7)])
cv2.imshow("Median", blurred_avg)
cv2.waitKey(0)

# Bilateral blurring
# In order to reduce noisewhile still maintaining edges, we can use bilateral blurring
# two Gaussian distributions.
