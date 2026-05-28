import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('0326/building.png')
rows,cols,ch=img.shape

pts1 = np.float32([[100,100],[300,100],[100,400]])
pts2 = np.float32([[45,100],[240,55],[100,250]])

M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img,M, (cols, rows))
dst_rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img_rgb), plt.title('Input')
plt.subplot(122), plt.imshow(dst_rgb), plt.title('Output')
plt.show()