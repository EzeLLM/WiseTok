import cv2 as cv
import numpy as np

#a이미지
img = cv.imread('0911.png')
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
print(kernel)

#c이미지
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

#e이미지
dilation = cv.dilate(opening, kernel, anchor=(-1, -1), iterations=1)
erosion = cv.erode(dilation, kernel, anchor=(-1, -1), iterations=1)

result = np.hstack([img,opening,erosion])
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()
