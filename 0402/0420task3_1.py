import cv2 as cv 

img=cv.imread('practice5_src/apples.jpg', cv.IMREAD_GRAYSCALE)

blurred=cv.blur(img, (3,3))
dst=cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles=cv.HoughCircles(blurred,cv.HOUGH_GRADIENT,1,150,param1=100,param2=73)

for i in circles[0]: 
    cv.circle(dst,(int(i[0]),int(i[1])),int(i[2]),(0,0,255),2)

cv.imshow('Original', img)
cv.imshow('apples detection',dst)  

cv.waitKey()
cv.destroyAllWindows()