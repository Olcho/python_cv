#canny edge detection
import cv2 as cv

img=cv.imread('practice5_src/apples.jpg')	

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny1=cv.Canny(gray,50,150)
canny2=cv.Canny(gray,100,200)

cv.imshow('Original',gray)
cv.imshow('Canny1',canny1)
cv.imshow('Canny2',canny2)

cv.waitKey()
cv.destroyAllWindows()
