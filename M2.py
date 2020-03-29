import numpy as np
import cv2
from imutils import contours

ref = cv2.imread("D:/College/Sem VI/CV/ocr_ref.png")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

(refCnts, _) = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}
# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
    # compute the bounding box for the digit, extract it, and resize
    # it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:(y + h), x:(x + w)]
    roi2 = cv2.resize(roi, (57, 88))
    
    # update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi
cv2.imshow('digit 0', digits[0])
cv2.waitKey()

img_path = 'D:/College/Sem VI/CV/cc2.jfif'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Image", gray)
cv2.waitKey(0)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
cv2.imshow('Sobel', gradX)
cv2.waitKey()

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow('Thresh', thresh)
cv2.waitKey()

(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# for cnt in cnts:
#     peri = cv2.arcLength(cnt, True)  # approx area: would return a rectangle
#     # print(peri)
#     approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approximate polynomials 0.04*peri = 4% of the approx area would mean better approximation of polynomial than peri
#     print(approx.shape)
#     thresh2 = img.copy()
#     cv2.drawContours(thresh2, cnt, -1, (255, 255, 255), 3)
#     # thresh2 = cv2.putText(thresh2, str(approx.shape), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
#     cv2.imshow("shape contours", thresh2)
#     cv2.waitKey(0)

locs = []
max = 0
for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # print(ar)
    # we can prune potential contour based on maximum aspect ratio as 16 digit number would have largest w:h ratio
    if ar > max:
        # contours can further be pruned on minimum/maximum width
        # and height
        # print(w, h)
        if (w > 200 and w < 250) and (h > 10 and h < 20):
            # append the bounding box region of the digits group
            # to our locations list
            locs.append((x, y, w, h))

(gX, gY, gW, gH) = locs[0]
cv2.rectangle(img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
cv2.imshow('Number', img)
cv2.waitKey()

# apply thresholding to segment the digits from the background of the credit card
num = gray[gY - 2:gY + gH + 2, gX - 2:gX + gW + 2]
num = cv2.threshold(num, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Number cropped', num)
cv2.waitKey()
# detect the contours of each individual digit in the group, then sort the digit contours from left to right
(digitCnts, _) = cv2.findContours(num.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
dcounts2 = []
for cnt in digitCnts:
    peri = cv2.arcLength(cnt, True)
    # print(peri)
    if peri > 5.0:
        dcounts2.append(cnt)
        # approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        # print(approx.shape)
        # thresh2 = num.copy()
        # cv2.drawContours(thresh2, cnt, -1, (255, 255, 255), 3)
        # # thresh2 = cv2.putText(thresh2, str(approx.shape), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        # cv2.imshow("shape contours", thresh2)
        # cv2.waitKey(0)

len(dcounts2)
# loop over the digit contours
pred = []
for c in dcounts2:
    # compute the bounding box of the individual digit, extract the digit, and resize it to have the same fixed size as the reference OCR-A images
    (x, y, w, h) = cv2.boundingRect(c)
    roi = num[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # cv2.imshow('w', roi)
    # cv2.waitKey()
    # initialize a list of template matching scores
    scores = []
    print(c)
    # loop over the reference digit name and digit ROI
    for (digit, digitROI) in digits.items():
        # apply correlation-based template matching, take the score, and update the scores list
        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        scores.append(score)
    
    # the classification for the digit ROI will be the reference digit name with the *largest* template matching score
    pred.append(str(np.argmax(scores)))
    # draw the digit classifications around the group
    cv2.rectangle(img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv2.putText(img, "".join(pred), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

cv2.imshow('OCR', img)
cv2.waitKey()
