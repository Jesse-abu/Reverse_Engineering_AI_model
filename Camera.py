import cv2
import numpy as np

cam = cv2.VideoCapture(0)
orb = cv2.ORB.create(nfeatures=100)

if not cam.isOpened():
    print('Failed to open camera')
    exit()



print('Exit button: q, picture button: r')

while True:
    retrieve, frame = cam.read()

    if not retrieve:
        print('failed to grab grame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    frame_with_features = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 9), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('live_feature_detector', frame_with_features)

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('r'):        
        pass

cam.release()
cv2.destroyAllWindows()
