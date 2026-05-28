import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('0326/building.png',cv.IMREAD_UNCHANGED) 

t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(bin_img,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()


b = bin_img[100:250, 100:300]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# structuring element
se=np.uint8([[0,0,1,0,0],
             [0,1,1,1,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
             [0,0,1,0,0]])

# dilation operation
b_dilation=cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# erosion operation
b_erosion=cv.erode(b, se, iterations=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# Write closing and opening operations #
b_closing=cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_opening=cv.dilate(cv.erode(b, se, iterations=1), se, iterations=1)
plt.imshow(b_opening, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
