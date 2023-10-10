import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[5]
#print(tracker_type)

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

#print(tracker)

video = cv2.VideoCapture('Videos/race.mp4')
if not video.isOpened():
    print('Error while loading the video!')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Erro while loading the frame!')
    sys.exit()
#print(ok)

bbox = cv2.selectROI(frame) # region of interest
print(bbox)

ok = tracker.init(frame, bbox)
print(ok)

colors = (randint(0, 255), randint(0,255), randint(0, 255)) # RGB -> BGR
print(colors)

while True:
    ok, frame = video.read()
    #print(ok)
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    #print(ok, bbox)
    if ok == True:
        (x, y, w, h) = [int(v) for v in bbox]
        #print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)
    else:
        cv2.putText(frame, 'Tracking failure!', (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255))

    cv2.putText(frame, tracker_type, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255))

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27: # esc
        break
























