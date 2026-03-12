#Trackbar example - adjust the brightness of image
import numpy as np
import cv2 as cv

def onChange(value):
    global img, title
    
    add_value = value - int(img[0][0])
    print("add_value:", add_value)
    img[:] = img + add_value
    cv.imshow(title, img)
    
img = np.zeros((300, 500), np.uint8) #create a black image
title = 'Trackbar example'
cv.imshow(title, img)

cv.createTrackbar('Trackbar test', title, img[0][0], 255, onChange)
cv.waitKey(0)
cv.destroyAllWindows()