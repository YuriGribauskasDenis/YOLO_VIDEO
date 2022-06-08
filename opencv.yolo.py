import cv2
import numpy as np
import random as rd
from pathlib import Path


def colors_func(n): 
    ret = []
    r = int(rd.random() * 256)
    g = int(rd.random() * 256)
    b = int(rd.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

def countCameras():
    """Returns count of found cameras (not opened yet).
    Brute force - just tries them all till it fails."""
    camCount = 0
    while True:
        cam = cv2.VideoCapture(camCount, cv2.CAP_DSHOW)
        if cam.isOpened():
            print("Found camera #", camCount)
            cam.release()
            camCount += 1
        else:
            return camCount


directory = Path(__file__).resolve().parent
resources = directory / 'RESOURCES'

print("OpenCV version:", cv2.__version__)
print("Total cameras found:", countCameras())

#cap = cv2.VideoCapture(0)
# This seems to be a bug in MSMF backend of opencv
# For windows platform, you can change the backend
# to something else (most preferably DirectShow backend)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #capture device - camera

classNames = []
classFile = resources / 'coco.names'
with open(classFile.__str__(), 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

N = len(classNames)
RGBcolors = colors_func(N)

modelConficuration =  resources / 'yolov3-tiny.cfg'
modelWeights =  resources / 'yolov3-tiny.weights'
whidthTarget = 320 #same as height according to model

confidenceThreshhold = 0.6
# the lower it is the more agressive is the nms method
nmsThreshhold = 0.3
def findObject(outputs,img):
    hT, wT, cT = img.shape
    #we chack a good detection and put it's result in next lists
    bbs = []
    classIds = []
    confs = []

    for output in outputs:
        #detection
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshhold:
                # they are the percentage of actual width and height
                # we multiply with real w&h
                # and they are pixel values so we delete floating point
                w,h = int(det[2]*wT),int(det[3]*hT)
                # convert cx&cy to corners x&y scaled to BB size
                # int just for luls
                x,y = int(det[0]*wT - w/2), int(det[1]*hT - h/2)
                bbs.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbs))
    # none maximum supprresion
    # remove overlaping boxes
    # pick max conf box and suppress all others
    keepIndeces = cv2.dnn.NMSBoxes(bbs,confs,confidenceThreshhold,nmsThreshhold)
    for i in keepIndeces:
        i = i[0]
        box = bbs[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        # print(classIds[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),RGBcolors[classIds[i]],2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {round(confs[i]*100,1)}%',
                    (x+5,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,RGBcolors[classIds[i]],2)

net = cv2.dnn.readNetFromDarknet(modelConficuration.__str__(),modelWeights.__str__())
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    _ , img = cap.read()
    # blob conversion, cause yolo takes blob as input
    blob = cv2.dnn.blobFromImage(img,1/255,(whidthTarget,whidthTarget),[0,0,0],1,False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    idxs = net.getUnconnectedOutLayers()
    # print(idxs)
    outputLayersNames = [layerNames[i[0]-1] for i in idxs]
    # print(outputLayersNames)

    outputs = net.forward(outputLayersNames)
    """
    #print(len(outputs))
    #print(type(outputs))
    #print(type(outputs[0]))
    #(300, 85)
    # 300 BBs, 85 values
    # 80 classes and 5 values.
    # 80 probabilities of each class
    # 4 arecenter x&y, width, height
    # 5 confidence that there is an object at all
    # cx,cy,w,h,confidence,person,bycicle,car,...,toothbrush
    # 0..299
    print(outputs[0].shape)
    #(1200, 85)
    print(outputs[1].shape)
    #(4800, 85)
    print(outputs[2].shape)
    # totally 6300 boxes, almost all with low confidence
    # we must run through all of ther
    #print(outputs[0][0])
    """
    findObject(outputs,img)

    cv2.imshow('YOLO NMS',img,)
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
    elif c == ord('c'):
        RGBcolors = colors_func(N)