import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f=open('w13/coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    model=cv.dnn.readNet('w13/yolov3.weights','w13/yolov3.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]		# box, confidence, class id
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# take only if confidence is more than 50%
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()	# YOLO model construction
colors=np.random.uniform(0,255,size=(100,3))		# colors for tracks

from sort import Sort

sort=Sort()

cap=cv.VideoCapture('w13/traffic3.mp4')
#cap=cv.VideoCapture(0,cv.CAP_DSHOW) # if webcam is used instead
if not cap.isOpened(): sys.exit('Camera connection failed')

cv.destroyAllWindows()
cv.namedWindow('Person tracking by SORT resized', cv.WINDOW_NORMAL)
cv.resizeWindow('Person tracking by SORT resized', 960, 540)

while True:
    ret,frame=cap.read()
    if not ret: sys.exit('Frame not captured')

    res=yolo_detect(frame,model,out_layers)   
    persons=[res[i] for i in range(len(res)) if res[i][5]==0] # class 0

    if len(persons)==0: 
        tracks=sort.update()
    else:
        tracks=sort.update(np.array(persons))
    
    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id=tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],10)
        cv.putText(frame,str(track_id),(x1+10,y1+40),
                   cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)
                    
    display = cv.resize(frame, (960, 540))
    cv.imshow('Person tracking by SORT',display)
    
    key=cv.waitKey(1) 
    if key==ord('q'): break 
    
cap.release()		# Camera disconnected
cv.destroyAllWindows()