import cv2 as cv

soccer = cv.imread('practice3_src/soccer.jpg')
rose = cv.imread('practice3_src/rose.png')

if soccer is None or rose is None:
    print('file not found')

#red channel-soccer
soccer_r = soccer[:, :, 2]

t1, th1 = cv.threshold(soccer_r, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print("soccer (red) threshold:", t1)

#grayscale-rose
rose_gray = cv.cvtColor(rose, cv.COLOR_BGR2GRAY)

t2, th2 = cv.threshold(rose_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print("rose (gray) threshold:", t2)

#rose-red channel
rose_r = rose[:, :, 2]

t3, th3 = cv.threshold(rose_r, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print("rose (red) threshold:", t3)

titles = ['soccer_red', 'rose_gray', 'rose_red']
images = [th1, th2, th3]

for i in range(3):
    cv.imshow(titles[i], images[i])

cv.waitKey()
cv.destroyAllWindows()