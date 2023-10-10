import cv2, sys, os
from random import randint

if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    print('Error loading the network files!')
    sys.exit()

tracker = cv2.TrackerGOTURN_create()

video = cv2.VideoCapture('Videos/race.mp4')
if not video.isOpened():
    print('Error while loading the video!')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Erro while loading the frame!')
    sys.exit()

bbox = cv2.selectROI(frame) # region of interest

ok = tracker.init(frame, bbox)

colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    if ok == True:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2)
    else:
        cv2.putText(frame, 'Tracking failure!', (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255))

    cv2.putText(frame, 'GOTURN', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255))

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27: # esc
        break


