# transfer learning example 1

import cv2 as cv 
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

model=EfficientNetB0(weights='imagenet')

img=cv.imread('w11/DSC_0507.jpg') 
x=np.reshape(cv.resize(img,(224,224)),(1,224,224,3))   
x=preprocess_input(x)

preds=model.predict(x)
top5=decode_predictions(preds,top=5)[0]
print('예측 결과:',top5)

img_show=cv.resize(img,(800,533))

for i in range(5):
    text=top5[i][1]+':'+str(round(float(top5[i][2]),4))
    cv.putText(img_show,text,(10,30+i*30),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

cv.imshow('Recognition result',img_show)
cv.waitKey()
cv.destroyAllWindows()