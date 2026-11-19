import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = "./pose_landmarker_full.task"
CAMERA_INDEX = 0

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode
PoseConnections = vision.PoseLandmarksConnections.POSE_LANDMARKS

# The result callback runs on a MediaPipe thread, so share the latest result via a lock.
latest_result = None
result_lock = threading.Lock()


def on_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result


def draw_pose(frame, result):
    """Draw landmarks and skeleton connections for every detected pose."""
    if result is None:
        return
    h, w = frame.shape[:2] # frame is a numpy array (h,w,c) => slicing [:2] returns h, w
    for landmarks in result.pose_landmarks: # results.pose_landmarks is an entry of a list of 33 landmarks per person
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks] # normalized landmarks are multiplied by w / h to convert into pixel coordinates
        for conn in PoseConnections: #list of landmark index pairs that form the skeleton, cv2.line draws line from start to end
            cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 0), 2)
        for p in points:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses=1,
    result_callback=on_result,
)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("error: could not open camera")
    raise SystemExit(1)
print("camera feed opened successfully, press 'q' to quit")

start = time.monotonic()
last_ts = -1

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("error: can't receive frame")
            break

        # MediaPipe expects RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly increasing.
        ts = int((time.monotonic() - start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        landmarker.detect_async(mp_image, ts)

        with result_lock:
            result = latest_result
        draw_pose(frame, result)
        #draw pose modifies the frame and returns nothing, that is why it
        #is called before displaying the frame

        cv2.imshow("webcam feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
