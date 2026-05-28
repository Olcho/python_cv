# Gamma correction
import cv2 as cv
import numpy as np

img=cv.imread('erica.jpg') 
img=cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25)

def gamma(f,gamma=1.0):
    return f

gc=np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),
              gamma(img,2.0),gamma(img,3.0)))
cv.imshow('gamma',gc)

cv.waitKey()
cv.destroyAllWindows()
