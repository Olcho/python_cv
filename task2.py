import cv2

img1 = cv2.imread("practice2_src/scene.jpg")
img2 = cv2.imread("practice2_src/Erica.jpg")

if img1 is None or img2 is None:
    print("File not found")
    exit()

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

window_name = "2024001130"

def onchange(a):
    alpha = a / 255.0
    result = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    cv2.imshow(window_name, result)

cv2.namedWindow(window_name)
cv2.createTrackbar("Dissolve", window_name, 0, 255, onchange)

onchange(0)

cv2.waitKey(0)
cv2.destroyAllWindows()