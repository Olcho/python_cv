import cv2 as cv 

img=cv.imread('coins.png', cv.IMREAD_GRAYSCALE)

blurred=cv.blur(img, (3,3))
dst=cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles=cv.HoughCircles(blurred,cv.HOUGH_GRADIENT,1,50,
                       param1=150,param2=30)

for i in circles[0]: 
    cv.circle(dst,(int(i[0]),int(i[1])),int(i[2]),(0,0,255),2)

cv.imshow('Original', img)
cv.imshow('Coin detection',dst)  

cv.waitKey()
cv.destroyAllWindows()
