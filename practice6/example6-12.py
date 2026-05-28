# -*- coding: utf-8 -*-
"""
OpenCV Harris Corner Detector
@author: HYU
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

C=cv.cornerHarris(img,3,3,0.04)
C_norm=cv.normalize(C,0,255,cv.NORM_MINMAX)

# non-maximum suppression
# TO DO: complete the code for non-maximum suppression (8-connectivity)
            
np.set_printoptions(precision=2)
print(C_norm)					
print(img)	