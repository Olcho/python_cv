import cv2 as cv
import numpy as np

img = cv.imread('practice3_src/Lenna.png', cv.IMREAD_GRAYSCALE)

if img is None:
    print('file not found')

img32 = np.float32(img)
noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

cv.randn(noise, 0, 10)
noiseimg1 = np.uint8(np.clip(cv.add(img32, noise), 0, 255))

cv.randn(noise, 0, 20)
noiseimg2 = np.uint8(np.clip(cv.add(img32, noise), 0, 255))

cv.randn(noise, 0, 30)
noiseimg3 = np.uint8(np.clip(cv.add(img32, noise), 0, 255))

cv.imshow('original', img)
cv.imshow('std=10', noiseimg1)
cv.imshow('std=20', noiseimg2)
cv.imshow('std=30', noiseimg3)

cv.waitKey()
cv.destroyAllWindows()