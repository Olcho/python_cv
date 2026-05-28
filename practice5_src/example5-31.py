import cv2 as cv 
import numpy as np

img=cv.imread('building.jpg', cv.IMREAD_GRAYSCALE)

canny1=cv.Canny(img,50,150)
lines=cv.HoughLines(canny1,1,np.pi/180.0,250)

dst=cv.cvtColor(canny1, cv.COLOR_GRAY2BGR)

alpha=1000

for i in range(lines.shape[0]):
    r=lines[i,0,0] 
    t=lines[i,0,1]
    cos_t=np.cos(t)
    sin_t=np.sin(t)
    x0=r*cos_t
    y0=r*sin_t
    
    pt1=np.array([np.round(x0+alpha*(-sin_t)), 
                  np.round(y0+alpha*(cos_t))])
    pt2=np.array([np.round(x0-alpha*(-sin_t)), 
                  np.round(y0-alpha*(cos_t))])
    ipt1=np.int16(pt1)
    ipt2=np.int16(pt2)
    cv.line(dst, ipt1, ipt2, (0,0,255), 2, cv.LINE_AA)

cv.imshow('Original',img)
cv.imshow('Canny1',dst)

cv.waitKey()
cv.destroyAllWindows()