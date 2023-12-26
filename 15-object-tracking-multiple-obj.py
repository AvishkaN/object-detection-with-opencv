import cv2
import imutils
import numpy as np


TrDict = {'csrt': cv2.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
        #  'boosting' : cv2.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
        #  'tld': cv2.TrackerTLD_create,
        #  'medianflow': cv2.TrackerMedianFlow_create,
        #  'mosse':cv2.TrackerMOSSE_create
          }

trackers = cv2.MultiTracker_create()

# tracker = TrDict['csrt']()


# v = cv2.VideoCapture(0)
# v = cv2.VideoCapture('data/mot.mp4')
v = cv2.VideoCapture('data/cars_on_highway.mp4')
# v = cv2.VideoCapture('data/pexels-christopher-schultz.mp4')


ret, frame = v.read()

# k = 4
# for i in range(k):
#     print('ret',ret)

#     # frame = imutils.resize(frame,width=600)
#     cv2.imshow('Frame',frame)
#     bbi = cv2.selectROI('Frame',frame)
#     tracker_i=TrDict['csrt']()
#     trackers.add(tracker_i,frame,bbi)


k = 2
for i in range(k):
    # frame=imutils.resize(frame,width=600)
    cv2.imshow('Frame',frame)
    bbi = cv2.selectROI('Frame',frame)
    tracker_i = TrDict['csrt']()
    trackers.add(tracker_i,frame,bbi)



baseDir=r'D:\works\python\compoter-vision-udemy\16-object-detection\frames'
frameNumber=2

while True:
    ret,frame=v.read()
    if not ret:
        break

    # frame=imutils.resize(frame,width=600)
    (success,boxes) = trackers.update(frame)
    # np.savetxt(baseDir+'/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
    frameNumber+=1
    for box in boxes:
    # if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,255,0),5)
    # frame=imutils.resize(frame,width=600)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break


v.release()
cv2.destroyAllWindows()
