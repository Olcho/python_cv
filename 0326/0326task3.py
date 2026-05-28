import cv2 as cv

img = cv.imread('0326/building.png')
patch = img[106:206, 241:341]
print(img.shape)

img = cv.rectangle(img, (241,106), (341,206), (0,0,255), 1)
cv.imshow('original', img) 

p1 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation= cv.INTER_NEAREST)
p2 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation= cv.INTER_LINEAR)
p3 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation= cv.INTER_CUBIC)

cv.imshow('nearest', p1)
cv.imshow('linear', p2)
cv.imshow('cubic', p3)

cv.waitKey()
cv.destroyAllWindows()