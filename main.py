import cv2 as cv
import time
import math
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def distbetw(p,q):
    diffx = p.x - q.x
    diffy = p.y - q.y
    diffz = p.z - q.z
    return math.sqrt(diffx**2 + diffy**2 + diffz**2)


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = vision.RunningMode.VIDEO,
    num_hands = 1
)

webcam = cv.VideoCapture(0)
start_time = time.monotonic()

with vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        if not webcam.isOpened():
            break

        ret, frame = webcam.read()
        if not ret:
            print("Frame is not loaded")
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        timestamp = int((time.monotonic()-start_time) *1000)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mpImage, timestamp)
        count = 0

        if result.hand_landmarks:
            hand1 = result.hand_landmarks[0]
            for i in range(5):
                if i ==0 :
                    d1 = distbetw(hand1[4],hand1[2])
                    d2 = distbetw(hand1[2],hand1[0])
                    d3 = distbetw(hand1[4],hand1[0])
                    d4 = distbetw(hand1[4],hand1[3])
                    d5 = distbetw(hand1[3],hand1[0])
                    d6 = distbetw(hand1[4],hand1[0])
                    d7 = distbetw(hand1[4],hand1[3])
                    d8 = distbetw(hand1[3],hand1[2])
                    d9 = distbetw(hand1[4],hand1[2])
                    d10 = distbetw(hand1[4],hand1[1])
                    d11 = distbetw(hand1[1],hand1[0])
                    d12 = distbetw(hand1[4],hand1[0])
                    
                else:
                    d1 = distbetw(hand1[4+i*4],hand1[1+i*4])
                    d2 = distbetw(hand1[1+i*4],hand1[0])
                    d3 = distbetw(hand1[4+i*4],hand1[0])
                    d4 = distbetw(hand1[4+i*4],hand1[3+i*4])
                    d5 = distbetw(hand1[3+i*4],hand1[0])
                    d6 = distbetw(hand1[4+i*4],hand1[0])
                    d7 = distbetw(hand1[4+i*4],hand1[3+i*4])
                    d8 = distbetw(hand1[3+i*4],hand1[2+i*4])
                    d9 = distbetw(hand1[4+i*4],hand1[2+i*4])
                    d10 = distbetw(hand1[4+i*4],hand1[2+i*4])
                    d11 = distbetw(hand1[2+i*4],hand1[0])
                    d12 = distbetw(hand1[4+i*4],hand1[0])

                if d3/(d1+d2) >= 0.9 and d6/(d4+d5) >= 0.9 and d9/(d7+d8) >= 0.9 and d12/(d10+d11) >= 0.9:
                    count +=1
        cv.putText(frame,f"Fingers: {count}",(int(frame.shape[1]*0.7),int(frame.shape[0]*0.3)),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
        cv.imshow("WebCam", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

webcam.release()
cv.destroyAllWindows()
print("Program Ended!")