import cv2 as cv
import numpy as np

img1 = cv.imread('practice6/mot_color70.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

rows, cols = gray1.shape
M = cv.getRotationMatrix2D((cols/2, rows/2), 30, 0.8)
gray2 = cv.warpAffine(gray1, M, (cols, rows))

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv.BFMatcher()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img_match = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:10], None, 
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Original(L) - Transformed(R)', img_match)
cv.waitKey(0)
cv.destroyAllWindows()