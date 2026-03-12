import cv2 as cv
import sys

img=cv.imread('Erica.jpg')

if img is None:
    print('File not found')

cv.imshow('Original', img)
cv.imshow('Upperleft', img[0:img.shape[0]//2, 0:img.shape[1]//2, :])
cv.imshow('CenterHalf', img[img.shape[0]//4:3*img.shape[0]//4,
                            img.shape[1]//4:3*img.shape[1]//4, :])

cv.imshow('R channel', img[:, :, 2])
cv.imshow('G channel', img[:, :, 1])
cv.imshow('B channel', img[:, :, 0])

cv.waitKey()
cv.destroyAllWindows()
