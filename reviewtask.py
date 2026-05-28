import cv2 as cv
import numpy as np

img = cv.imread('practice3_src/Lenna.png', cv.IMREAD_GRAYSCALE)

if img is None:
    print('file not found')

bImg = cv. blur(img, (5,5))

sumimg = cv.integral(img)
bImg2 = np.zeros((img.shape[0], img.shape[1]))

h, w = img.shape

for y in range(h):

    for x in range(w):
        y1 = max(0, y - 2)
        y2 = min(h - 1, y + 2)

        x1 = max(0, x - 2)
        x2 = min(w - 1, x + 2)

        area_sum = sumimg[y2 + 1, x2 + 1] - sumimg[y1, x2 + 1] - sumimg[y2 + 1, x1] + sumimg[y1, x1]
        area = (y2 - y1 + 1) * (x2 - x1 + 1)
        bImg2[y, x] = area_sum / area

bImg2 = bImg2.astype(np.uint8) #uint8타입변환

titiles = ['original Image', 'Blurred', 'with IntegralImg']
images = [img, bImg, bImg2]

for i in range(3):
    cv.imshow(titiles[i], images[i])

cv.waitKey()
cv.destroyAllWindows()
