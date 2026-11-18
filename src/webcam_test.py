import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror view; remove this line for unmirrored
    cv2.imshow("webcam", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  #q or Esc
        break

cap.release()
cv2.destroyAllWindows()
