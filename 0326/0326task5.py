import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('0326/paper.png')
rows,cols,ch=img.shape
pts1 = np.float32([[472, 71],[736, 107],[371, 446],[707, 515]])
pts2 = np.float32([[0,0],[399,0],[0,371],[399,371]])

M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(399,371))

upts1 = np.uint16(pts1)
for i in range(4):
    cv.circle(img,(upts1[i,0],upts1[i,1]),8,(0,255,0),-1)

dst_rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img_rgb),plt.title('Input')
plt.subplot(122),plt.imshow(dst_rgb),plt.title('Output')
plt.show()