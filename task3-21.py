import cv2 as cv

img = cv.imread('stddev30.jpg', cv.IMREAD_GRAYSCALE)
if img is None:
    print('file not found')

gImg1 = cv.GaussianBlur(img, (5, 5), 0)
gImg2 = cv.GaussianBlur(img, (11, 11), 0)
gImg3 = cv.GaussianBlur(img, (23, 23), 0)
gImg4 = cv.GaussianBlur(img, (45, 45), 0)

titles =['original', 'GaussianBlur5', 'GaussianBlur11', 'GaussianBlur23', 'GaussianBlur45']
images = [img, gImg1, gImg2, gImg3, gImg4]

for i in range(5):
    cv.imshow(titles[i], images[i])

cv.waitKey()
cv.destroyAllWindows()