import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('checkerboardWithDots.png')
rows,cols,ch=img.shape

pts1=np.float32([[100,100],[300,100],[100,400]])
pts2=np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)
dst=cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
