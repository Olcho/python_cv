import cv2 as cv
import numpy as np

img = cv.imread('mot_color70.jpg')
gray = cv.cvtcolor(img, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()

kp, des = orb.detectAndCompute(gray, None)

print("descriptor size : ", orb.descriptorSize())
print("descriptor shape : ", des.shape())

gray_result = cv.drawKeypoints(gray, kp, None, color=(0,255,0),
                               flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('ORB Keypoint Detection', gray_result)
cv.waitKey(0)
cv.destroyAllWindows()