# -*- coding: utf-8 -*-
"""
FAST Corner Detection

@author:
"""

import cv2 as cv

img=cv.imread('building.jpg', cv.IMREAD_GRAYSCALE)

# Initiate FAST object detector with default values
fast=cv.FastFeatureDetector_create(threshold=60)

# Find and draw the keypoints
kp=fast.detect(img,None)
img2=cv.drawKeypoints(img, kp, None, color=(0,0,255))

# Print all default parameters
print("Threshold:{}".format(fast.getThreshold()))
print("NonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("Neighborhood:{}".format(fast.getType()))
print("Total keypoints with nonmaxSuppression:{}".format(len(kp)))

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp=fast.detect(img,None)

print("Total keypoints with nonmaxSuppression:{}".format(len(kp)))
img3=cv.drawKeypoints(img, kp, None, color=(0,0,255))

cv.imshow('With non-maximum suppression', img2)
cv.imshow('Without non-maximum suppression', img3)
cv.waitKey()
cv.destroyAllWindows()