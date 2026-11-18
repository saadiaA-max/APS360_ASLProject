import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    raise SystemExit

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB; OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)  # must keep increasing
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # landmarks are fractions of width/height, so scale to pixels
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for lm in result.hand_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        cv2.imshow("landmarks", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or Esc
            break

cap.release()
cv2.destroyAllWindows()