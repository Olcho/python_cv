"""
 Harris corner detection algorithm
 @author:
"""

import cv2 as cv
import numpy as np

img=np.array([[0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0,0,0],
              [0,0,0,1,1,1,0,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,1,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)

ux=np.array([[-1,0,1]])
uy=np.array([-1,0,1]).transpose()
k=cv.getGaussianKernel(3,1) # k: a 3x1 Gaussian mask
g=np.outer(k,k.transpose()) # g: a 3x3 Gaussian mask

# Step 1. 
dy = cv.filter2D(img, -1, uy)
dx = cv.filter2D(img, -1, ux)

# Step 2. 
dyy = dy*dy
dxx = dx*dx
dyx = dy*dx

# Step 3.
gdyy = cv.filter2D(dyy, -1, g)
gdxx = cv.filter2D(dxx, -1, g)
gdyx = cv.filter2D(dyx, -1 ,g)

# Step 4. 
det_A = (gdxx * gdyy) - (gdyx ** 2)
trace_A = gdxx + gdyy
k_val = 0.04
C = det_A - k_val * (trace_A ** 2)

# Step 5. 
corners = np.zeros(C.shape, dtype=np.uint8)
for j in range(1, C.shape[0]-1):
    for i in range(1, C.shape[1]-1):
        if C[j, i] > 0.1:
            neighbor = C[j-1:j+2, i-1:i+2]
            if C[j, i] == np.max(neighbor):
                corners[j, i] = 1					

# for displaying
popping=np.zeros([160,160],np.uint8)  
for j in range(0,160):
   for i in range(0,160):
       popping[j,i]=np.uint8(np.clip((C[j//16, i//16] + 0.06) * 700, 0, 255))

cv.imshow('Harris Corner Detection Result', popping)    
cv.waitKey(0)
cv.destroyAllWindows()