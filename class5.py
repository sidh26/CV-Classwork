import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = 'D:/Downloads/barcode2.jpeg'
img = cv2.imread(img_path, 0)
cv2.imshow("Image", img)
cv2.waitKey(0)

# Erosion builds a kernel moving all over the image
# if all surrounding pixels are while then pixel will be white otherwise it will be black
# Unbolden
# useful for removing small white noises, detach two connected objects etc.
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow("Erosion", erosion)
cv2.waitKey(0)

# Dilation: opposite of erosion
# if one surrounding pixel is white then pixel is white
# increase whiteness in region of interest
# Bolden
# Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it.
# Since noise is gone, they won't come back, but our object area increases.
# It is also useful in joining broken parts of an object.
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)

# Opening: erosion followed by dilation
# "neat" thing close to median filter
# open any white spots and fill with black

# Closing: dilation followed by erosion
# Close black spots
# useful in closing small holes inside the foreground objects, or small black points on the object
# Useful while contouring

# Morphological Gradient: difference between dilation and erosion
# Useless

# Top hat and black hat
# Not used a lot

# Structuring element
# Flexibility to use smaller and bigger kernels and do operations
# Not necessarily square or rect

# Detect Barcode
img_path = 'D:/Downloads/barcode3.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", img)
cv2.waitKey(0)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
# subtract the y-gradient from the x-gradient
# Fill gaps and blur
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
# get rid of small mag gradients
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
# fill gap using morph trans
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 55))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# sequentially erode multiple times
closed = cv2.erode(closed, None, iterations=20)  # fill white spots
closed = cv2.dilate(closed, None, iterations=25)  # fill black spots
# cv2.imshow("Image", closed)
# cv2.waitKey(0)
# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
cv2.imshow("Barcode", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# now crop the barcode and send to scanning software

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


finimg = four_point_transform(img, box)
cv2.imshow('cropped', finimg)
cv2.waitKey(0)
