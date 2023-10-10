import cv2
import sys
from random import randint

tracker = cv2.legacy.TrackerCSRT_create()

video = cv2.VideoCapture('Videos/walking.avi')
if not video.isOpened():
    print('Error while loading the video!')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Erro while loading the frame!')
    sys.exit()

cascade = cv2.CascadeClassifier('cascade/fullbody.xml')

def detect():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(frame_gray, minSize=(60,60))
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.imshow('Detection', frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            if x > 0:
                print('Haarscade detection')
                return x, y, w, h

#bbox = detect()
bbox = cv2.selectROI(frame)
#print(bbox)

ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors)
    else:
        print('Tracking failure! We will execute the haarcascade detector')
        bbox = detect()
        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, bbox)

    cv2.imshow('Tracking', frame)
    k = cv2.waitKey(1) & 0XFF
    if k == 27: # esc
        break

















