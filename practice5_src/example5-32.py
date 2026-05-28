import cv2 as cv 
import numpy as np

img=cv.imread('building.jpg', cv.IMREAD_GRAYSCALE)

canny1=cv.Canny(img,50,200)
lines=cv.HoughLinesP(canny1,1, np.pi/180.0,80,30,10)

dst=cv.cvtColor(canny1, cv.COLOR_GRAY2BGR)

for i in range(lines.shape[0]):
    pt = lines[i]
    cv.line(dst, (pt[0,0], pt[0,1]), (pt[0,2], pt[0,3]), 
            (0,0,255), 2, cv.LINE_AA)

cv.imshow('Original',img)
cv.imshow('Canny1',dst)

cv.waitKey()
cv.destroyAllWindows()
