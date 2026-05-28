import cv2 as cv
import numpy as np

img = cv.imread('practice5_src/apples.jpg', cv.IMREAD_GRAYSCALE)

grad_x = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)

#Q1
fmag = cv.magnitude (grad_x, grad_y)
mag = np.uint8(fmag)

target_y, target_x = 775, 377
magnitude_value = fmag[target_y, target_x]

#Q2
max_val = np.max(fmag)
y_max, x_max = np.unravel_index(fmag.argmax(), fmag.shape)

#Q3
fanlgle = cv.phase(grad_x, grad_y, angleInDegrees=True)
target_y, target_x = 775, 377
grad_dir = fanlgle[target_y, target_x]
edge_dir = (grad_dir + 90) %360

#Q1 print
print(f"({target_x}, {target_y}) edge magnitude : {magnitude_value}")

#Q2 print
print(f"max val: {max_val}")
print(f"max val pixel(x, y): ({x_max}, {y_max})")

#Q3 print
print(f"grad_dir : {grad_dir:.2f}")
print(f"edge_dir : {edge_dir:.2f}")

cv.imshow('mag',mag)
cv.waitKey()
cv.destroyAllWindows()