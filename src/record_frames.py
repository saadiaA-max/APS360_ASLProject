import argparse
import os
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--signer", required=True)
parser.add_argument("--letters", nargs="+", required=True)
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--interval", type=float, default=0.2)  # seconds between saves
args = parser.parse_args()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    raise SystemExit

quit_all = False
for letter in args.letters:
    if quit_all:
        break
    out_dir = os.path.join("data", "raw", args.signer, letter)
    os.makedirs(out_dir, exist_ok=True)

    #wait until SPACE is pressed (Esc quits)
    while True:
        ok, frame = cap.read()
        if not ok:
            quit_all = True
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Sign {letter} - press SPACE to start", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("record", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break
        if key == 27:
            quit_all = True
            break
    if quit_all:
        break

    #record frames for this letter
    count = 0
    last_save = 0.0
    while count < args.frames:
        ok, frame = cap.read()
        if not ok:
            quit_all = True
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        if now - last_save >= args.interval:
            count += 1
            cv2.imwrite(os.path.join(out_dir, f"{count:04d}.jpg"), frame)  # clean frame, no text
            last_save = now
        display = frame.copy()
        cv2.putText(display, f"{letter}: {count}/{args.frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("record", display)
        if cv2.waitKey(1) & 0xFF == 27:
            quit_all = True
            break

cap.release()
cv2.destroyAllWindows()