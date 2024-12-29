import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fontScale = 3

font = cv2.FONT_HERSHEY_PLAIN
classNames = []
classFiles = 'C:\\Users\\DELL\\Desktop\\yash\\python\\cv\\objectDetection\\coco.names'
with open(classFiles,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'objectDetection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'objectDetection\\frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(weightsPath,configPath) #A DNN (Deep Neural Network) object detection model loaded using OpenCV's cv2.dnn_DetectionModel.
model.setInputSize(320,320) # size of new frame
model.setInputScale(1.0/127.5) # scale factor ,Normalizes pixel values by scaling 
model.setInputMean((127.5,127.5,127.5)) # mean val of frame,  Subtracts the mean values (127.5, 127.5, 127.5) from each pixel channel for normalization.
model.setInputSwapRB(True)
font_scale = 3
while(True):  
    ret, frame = cap.read()
    classIndex, confidence, bbox = model.detect(frame,confThreshold= 0.55)

    if(len(classIndex)!= 0):
        for classInd, conf,boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
            if(classInd<=30):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classNames[classInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale = font_scale, color=(0,255,255),thickness=3)
    cv2.imshow('obj detection',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWIndows()
