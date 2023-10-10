import cv2
import time
from imutils.video import VideoStream

cap = VideoStream(src=0).start()
time.sleep(1.0)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
#print(ok)

bbox = cv2.selectROI(frame)
x, y, w, h = bbox
track_window = (x, y, w, h)
print(track_window)

roi = frame[y:y+h, x:x+w] # RGB -> BGR
#cv2.imshow('ROI', roi)
#cv2.waitKey(0)

# HSV
# https://cran.r-project.org/web/packages/colordistance/vignettes/color-spaces.html
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#cv2.imshow('ROI HSV', hsv_roi)
#cv2.waitKey(0)

# https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])

import matplotlib.pyplot as plt
plt.hist(roi.ravel(), 180, [0, 180])
#plt.show()
#cv2.waitKey(0)

# 0 - 255
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

parameters = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ok, frame = cap.read()
    if ok == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # https://docs.opencv.org/3.4.15/da/d7f/tutorial_back_projection.html
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ok, track_window = cv2.meanShift(dst, (x, y, w, h), parameters)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Meanshift tracking', frame)
        cv2.imshow('dst', dst)
        cv2.imshow('ROI', roi)

        if cv2.waitKey(1) == 13: # esc
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()




















