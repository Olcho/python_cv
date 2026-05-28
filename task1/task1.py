import cv2
import numpy as np

img = cv2.imread("practice2_src/Erica.jpg")
if img is None:
    print("File not found")
    exit()

img_reduce = cv2.resize(img, None, fx=0.5, fy=0.5)

hsv = cv2.cvtColor(img_reduce, cv2.COLOR_BGR2HSV)
cv2.waitKey(0)
cv2.destroyAllWindows()