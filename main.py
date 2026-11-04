import cv2 
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "./pose_landmarker_full.task"
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

#Create a pose landmarker instance in live stream mode
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

#opencv2 video capture
camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened(): 
    print(f"error: could not open")
    exit()

print("camera feed opened successfully")

while True: 
    ret, frame = cap.read()

    if not ret: 
        print("Error: cant recieve frame")
        break

    cv2.imshow('webcam feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()