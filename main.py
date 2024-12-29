import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('C:\\Users\\DELL\\Desktop\\yash\\python\\cv\\objectDetection\\pfp_.jpeg')


classNames = []
classFiles = 'C:\\Users\\DELL\\Desktop\\yash\\python\\cv\\objectDetection\\coco.names'
with open(classFiles,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'objectDetection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'objectDetection\\frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(weightsPath,configPath)
model.setInputSize(320,320) # size of new frame
model.setInputScale(1.0/127.5) # scale factor 
model.setInputMean((127.5,127.5,127.5)) # mean val of frame
model.setInputSwapRB(True)


classIndex, confidence,bbox = model.detect(img,confThreshold= 0.5)
print(classIndex)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classNames[classInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale = font_scale,color=(0,255,0),thickness=3)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
cv2.imshow('output',img)
cv2.waitKey(0)