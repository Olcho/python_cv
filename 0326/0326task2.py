import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv.imread('0326/building.png')   # 다른 이미지로 바꿔도 됨
if img is None:
    print('file not found')
    exit()

# OpenCV는 BGR이라서 matplotlib 출력용으로 RGB 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

rows, cols, ch = img.shape

# 1. Scaling
scaled = cv.resize(img, dsize=None, fx=0.7, fy=0.7, interpolation=cv.INTER_LINEAR)
scaled_rgb = cv.cvtColor(scaled, cv.COLOR_BGR2RGB)

# 2. Translation
M_trans = np.float32([[1, 0, 80],
                      [0, 1, 50]])
translated = cv.warpAffine(img, M_trans, (cols, rows))
translated_rgb = cv.cvtColor(translated, cv.COLOR_BGR2RGB)

# 3. Rotation
M_rot = cv.getRotationMatrix2D((cols/2, rows/2), 30, 1.0)  # 중심 기준 30도 회전
rotated = cv.warpAffine(img, M_rot, (cols, rows))
rotated_rgb = cv.cvtColor(rotated, cv.COLOR_BGR2RGB)

# 4. Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[80, 70], [220, 30], [100, 220]])
M_aff = cv.getAffineTransform(pts1, pts2)
affine = cv.warpAffine(img, M_aff, (cols, rows))
affine_rgb = cv.cvtColor(affine, cv.COLOR_BGR2RGB)

# 5. Perspective Transformation
pts1_p = np.float32([[50, 50], [cols-50, 50], [50, rows-50], [cols-50, rows-50]])
pts2_p = np.float32([[100, 30], [cols-100, 80], [50, rows-80], [cols-30, rows-30]])
M_per = cv.getPerspectiveTransform(pts1_p, pts2_p)
perspective = cv.warpPerspective(img, M_per, (cols, rows))
perspective_rgb = cv.cvtColor(perspective, cv.COLOR_BGR2RGB)

# 결과 출력
titles = ['Original', 'Scaling', 'Translation', 'Rotation', 'Affine', 'Perspective']
images = [img_rgb, scaled_rgb, translated_rgb, rotated_rgb, affine_rgb, perspective_rgb]

plt.figure(figsize=(14, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()