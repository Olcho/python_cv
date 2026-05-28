import cv2 as cv

import random

img = cv.imread('practice3_src/Lenna.png', cv.IMREAD_GRAYSCALE)
if img is None:
    print('File not found')

noiseNum = img.size//10

for i in range(noiseNum):
    row = random.randrange(img.shape[0])
    col = random.randrange(img.shape[1])
    img[row,col] = (i % 2) * 255

cv.imshow('Salt and pepper noise', img)

#Gaussian Filter
gimg = cv.GaussianBlur(img, (5,5), 0)
cv.imshow('Median Filter', gimg)

#Median Filter
mimg = cv.medianBlur(img, 5)
cv.imshow('Median Filter', mimg)

cv.waitKey()
cv.destroyAllWindows()