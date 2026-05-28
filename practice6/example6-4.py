import cv2 as cv

img = cv.imread('practice6/mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# For BRIEF
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(gray,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray, kp)

print( "brief.descriptorSize : ", brief.descriptorSize() )
print( "des.shape : ", des.shape )

gray=cv.drawKeypoints(gray,kp,None,
                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('brief', gray)

# For ORB
# TO DO: Keypoint detection and description using ORB descriptor

orb = cv.ORB_create(patchSize=15)
kp_orb, des_orb = orb.detectAndCompute(gray, None)

print("ORB descriptor size:", orb.descriptorSize())
print("ORB descriptor shape:", des_orb.shape)

gray_orb = cv.drawKeypoints(gray, kp_orb, None, color=(0, 255, 0),
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('ORB Result', gray_orb)
k=cv.waitKey()
cv.destroyAllWindows()