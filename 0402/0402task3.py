import cv2 as cv 
import numpy as np

img=cv.imread('practice5_src/apples.jpg', cv.IMREAD_GRAYSCALE)

blurred=cv.blur(img, (3,3))
dst=cv.cvtColor(img, cv.COLOR_GRAY2BGR)

found_circles = None
target_count = 3
found_circles = None
best_p1, best_p2 = 0, 0

def find_optimal_circles():
    global found_circles, best_p1, best_p2

    for p1 in range(180, 99, -10):
        for p2 in range(200, 19, -5):
            circles = cv.HoughCircles(
                blurred, cv.HOUGH_GRADIENT, 1, 350, param1=p1, param2=p2, minRadius=30, maxRadius=120
            )
            if circles is not None:
                count = circles.shape[1]
                if count >= target_count:
                    found_circles = circles
                    best_p1, best_p2 = p1, p2 
                    return

find_optimal_circles()

if found_circles is not None:
    print(f"optimal parameter-> param1: {best_p1}, param2: {best_p2}")
    img_result = img.copy()
    circles = np.uint16(np.around(found_circles))
    
    # 가장 확실한 상위 3개만 그리기
    for i in circles [0, :target_count]:
        cv.circle(img_result, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv.circle(img_result, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('apples detection',img_result)
else:
    print("조건에 맞는 원을 찾지 못했습니다. p1, p2 범위를 확인하세요.")
cv.waitKey()
cv.destroyAllWindows()