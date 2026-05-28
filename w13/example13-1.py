import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img,flow,step=16):
    for y in range(step//2,frame.shape[0],step):
        for x in range(step//2,frame.shape[1],step):
            dx,dy=flow[y,x].astype(np.int32)
            if(dx*dx+dy*dy)>1: # red lines for large motions
                cv.line(img,(x,y),(x+dx,y+dy),(0,0,255),2) 
            else:
                cv.line(img,(x,y),(x+dx,y+dy),(0,255,0),2)            
    
cap=cv.VideoCapture(0,cv.CAP_DSHOW)	
if not cap.isOpened(): sys.exit('Camera connection failed.')
    
prev=None

while(1):
    ret,frame=cap.read()	# Capture a frame from the camera
    if not ret: sys('Frame not captured.')
    
    if prev is None:	# Only save the frame for the first frame
        prev=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        continue
    
    curr=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    flow=cv.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)
    
    draw_OpticalFlow(frame,flow)
    cv.imshow('Optical flow',frame)

    prev=curr

    key=cv.waitKey(1)	
    if key==ord('q'):	# Escape the loop when 'q' key pressed
        break 
    elif key==ord('s'): # Save the current frame to file
        cv.imwrite('OpticalFlow.jpg', frame)
    
cap.release()			# Disconnect the camera
cv.destroyAllWindows() 
