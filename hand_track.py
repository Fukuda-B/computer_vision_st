import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(2)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
window_x, window_y = 1280, 720
bp = cv2.imread('b.png')
bpGR = cv2.cvtColor(bp, cv2.COLOR_RGB2GRAY)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (window_x, window_y))
    imgRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgGR = cv2.cvtColor(imgS, cv2.COLOR_RGB2GRAY)
    imgHOT = cv2.applyColorMap(imgGR, cv2.COLORMAP_COOL)
    imgHOTS = cv2.resize(imgHOT, (int(window_x/4), int(window_y/4)))
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # hands info

    if results.multi_hand_landmarks: # if there is multiple hands
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(imgS, handLms, mpHands.HAND_CONNECTIONS) # single hand
            c0_pos = [0, 0] # tmp c0 pos
            for l_id, lm in enumerate(handLms.landmark): # l_id = landmark id
                # print(l_id, lm)
                h, w, c = imgS.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # position
                # print(cx, cy) # disp pos
                if l_id == 0 or l_id == 9: # xy rotation
                    cv2.circle(imgS, (cx, cy), 7, (255, 0, 255))
                    cv2.line(imgS, (cx, 0), (cx, window_y), (50, 50, 50))
                    cv2.line(imgS, (0, cy), (window_x, cy), (50, 50, 50))
                    cv2.putText(imgS, str(l_id)+'@'+str(cx)+'.'+str(cy), (cx-100, cy-10), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 50), 1)
                    if l_id == 0:
                        c0_pos = [cx, cy]
                    else:
                        cv2.circle(imgS, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                        cv2.line(imgS, (c0_pos[0], c0_pos[1]), (cx, cy), (0, 0, 255))
                        cv2.line(imgS, (cx, c0_pos[1]), (cx, cy), (255, 0, 0))
                        deg = '{:.6g}'.format(math.degrees(math.atan2(c0_pos[0]-cx, c0_pos[1]-cy)))
                        # print('deg:'+str(deg))
                        cv2.putText(imgS, 'atan(deg):'+str(deg), (cx-158, cy-25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 100), 1)
                elif l_id % 4 == 0:
                    cv2.circle(imgS, (cx, cy), 3, (0, 200, 0), cv2.FILLED)
                    cv2.putText(imgS, str(cx)+'.'+str(cy), (cx-80, cy-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 0), 1)
                    # cv2.drawMarker(imgS, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_DIAMOND, markerSize=5)

    mc = cv2.matchTemplate(imgGR, bpGR, cv2.TM_CCOEFF_NORMED) # zero-mean normalized cross correlation
    min_val, max_val, min_pt, max_pt = cv2.minMaxLoc(mc)
    print(f'{min_val} {max_val} {min_pt} {max_pt}')
    if max_val >= 0.82:
        pt = max_pt
        cv2.putText(imgS, 'sim_v:'+str('{:.6g}'.format(max_val)), (pt[0]-10, pt[1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.rectangle(imgS, (pt[0], pt[1]), (pt[0] + 50, pt[1] + 50), (0, 0, 255), 2)

    imgS[window_y-imgHOTS.shape[0]:window_y, 0:imgHOTS.shape[1]] = imgHOTS

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(imgS, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1) # fps

    cv2.imshow("B tracker", imgS)
    cv2.waitKey(1)

