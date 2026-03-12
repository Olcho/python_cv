import cv2
import numpy as np

img = cv2.imread("practice2_src/Erica.jpg")
img_reduce = cv2.resize(img, None, fx=0.5, fy=0.5)

hsv = cv2.cvtColor(img_reduce, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

h_new = (h.astype(np.int16) + 90) % 180
h_new = h_new.astype(np.uint8)

s_new = np.clip(s.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)

v_new = np.clip(v.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

hsv_final = cv2.merge([h_new, s_new, v_new])
result_img = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)


top = np.hstack((img_reduce, result_img))

combined = np.vstack((top, top))

cv2.imwrite("erica_new1.jpg", result_img)
cv2.imshow("2024001130", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()