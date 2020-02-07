import cv2
import numpy as np

img_path = 'D:\Jatayu Unmanned Systems\dataset2\Elon.jpg'
image = cv2.imread(img_path)
img = image
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))
cv2.imshow('Image', image)
cv2.waitKey(0)

# Important to note that OpenCV stores RGB channels in reverse order
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Split image into channels
(B, G, R) = cv2.split(image)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
cv2.waitKey(0)

# Extract image corner
corner = image[0:100, 0:300]
cv2.imshow("Corner", corner)
cv2.waitKey(0)

# Modify rectangular portion of image
img[150:200, 150:250] = (0, 255, 0)
cv2.imshow("Updated", img)
cv2.waitKey(0)

# Shapes on canvas
black = np.zeros((300, 300, 3))
white = np.full((300, 300, 3), 255.0)
cv2.imshow("Black", black)
cv2.imshow("White", white)
cv2.waitKey(0)

# Add lines
green = (0, 255, 0)
cv2.line(black, (0, 0), (300, 300), green)
cv2.imshow("Canvas", black)
cv2.waitKey(0)
red = (0, 0, 255)
cv2.line(black, (300, 0), (0, 300), red, 3)
cv2.imshow("Canvas", black)
cv2.waitKey(0)

# Add rectangles
cv2.rectangle(black, (10, 10), (60, 60), (255, 0, 0), thickness=cv2.FILLED)
cv2.imshow("Canvas", black)
cv2.waitKey(0)


# Function to add rectangle to image
def func_rect(img=None, h=None, w=None):
    if img is None:
        img = np.zeros((300, 400, 3))
    if h is None or w is None:
        (h, w, k) = img.shape
        h = int(h / 2)
        w = int(w / 2)
    print(h, w)
    cv2.rectangle(img, (h, h), (w, w), (0, 255, 0), cv2.FILLED)
    cv2.imshow('rect', img)
    cv2.waitKey(0)


func_rect()
func_rect(img)

# Adding circles to image
canvas = np.zeros((300, 300, 3), dtype="uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)
for r in range(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), r, white)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# Image Translation
M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted Down and Right", shifted)
cv2.waitKey(0)
M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted Up and Left", shifted)
cv2.waitKey(0)

# Image Rotation
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)
cv2.waitKey(0)

# Image resize
r = 150.0 / img.shape[1]
dim = (150, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)
cv2.waitKey(0)

# Image flipp
flipped = cv2.flip(img, 1)
cv2.imshow("Flipped Horizontally", flipped)
flipped = cv2.flip(img, 0)
cv2.imshow("Flipped Vertically", flipped)
flipped = cv2.flip(img, -1)
cv2.imshow("Flipped Horizontally and Vertically", flipped)
cv2.waitKey(0)

# Cropping
cropped = image[128:384, 128:384]
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)

cv2.destroyAllWindows()
