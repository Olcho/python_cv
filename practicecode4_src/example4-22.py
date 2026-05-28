import cv2 as cv
import numpy as np

img=cv.imread('tekapo.bmp')
rows, cols,channels = img.shape

M = np.float32([[1,0,100], [0,1,50]])
dst = cv.warpAffine(img, M, (cols, rows))

cv.imshow('Original', img)
cv.imshow('Translation', dst)

cv.waitKey()
cv.destroyAllWindows()