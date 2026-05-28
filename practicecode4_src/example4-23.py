import cv2 as cv

img=cv.imread('tekapo.bmp')
rows, cols,channels = img.shape

M=cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 
                         30, 1)
dst = cv.warpAffine(img, M, (cols, rows))

cv.imshow('Original', img)
cv.imshow('Rotation', dst)

cv.waitKey()
cv.destroyAllWindows()
