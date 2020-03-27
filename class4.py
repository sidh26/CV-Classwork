import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = 'D:/Downloads/triangle.png'
img = cv2.imread(img_path, 0)

# do either thresholding or canny edge detection

# Thresholding
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
cv2.imshow("Mean Thresh", thresh)
cv2.waitKey(0)


# canny edge detection
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


canny = auto_canny(img, 0.75)
cv2.imshow("Auto Canny", canny)
cv2.waitKey(0)

# Find contours on thresh or canny
cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(cnts)  # all points on the contours, raw contour
peri = cv2.arcLength(cnts[0], True)  # approx area: would return a rectangle
print(peri)
approx = cv2.approxPolyDP(cnts[0], 0.04 * peri, True)  # approximate polynomials 0.04*peri = 4% of the approx area would mean better approximation of polynomial than peri
print(approx)
img.shape
black = np.zeros((920, 674, 3))
white = [255, 255, 255]
black[465, 121] = white
black[225, 600] = white
black[708, 600] = white
black[467, 121] = white
black[708, 599] = white
black[225, 599] = white

cv2.imshow("approx poly", black)
cv2.waitKey(0)

shapes = cv2.imread('D:/Downloads/shapes_1.png')
shapes = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(shapes, 5)
cv2.imshow("Median", median)
cv2.waitKey(0)
thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)  # approx area: would return a rectangle
    # print(peri)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approximate polynomials 0.04*peri = 4% of the approx area would mean better approximation of polynomial than peri
    print(approx.shape)
    thresh2 = thresh.copy()
    cv2.drawContours(thresh2, cnt, -1, (0, 0, 0), 3)
    # thresh2 = cv2.putText(thresh2, str(approx.shape), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imshow("shape contours", thresh2)
    cv2.waitKey(0)

cv2.destroyAllWindows()

contours = shapes.copy()
cv2.drawContours(contours, cnts, -1, (0, 0, 255), 2)
cv2.imshow("shape contours", contours)
cv2.waitKey(0)

maxPeri = 0
numPoints = 0
for c in range(len(cnts)):
    peri = cv2.arcLength(cnts[c], True)
    if peri > maxPeri:
        maxPeri = peri
        numPoints = cv2.approxPolyDP(cnts[c], 0.04 * peri, True)

print(maxPeri)
print(numPoints)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


numPoints1 = np.reshape(numPoints, (4, 2))
temp = order_points(numPoints1)


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    # print(rect)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


finimg = four_point_transform(thresh2, temp)
cv2.imshow(finimg)
cv2.waitKey()
