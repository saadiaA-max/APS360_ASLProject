import argparse
import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

parser = argparse.ArgumentParser()
parser.add_argument("--signer", required=True)
parser.add_argument("--size", type=int, default=64)
parser.add_argument("--margin", type=float, default=0.25)
args = parser.parse_args()

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)

raw_dir = os.path.join("data", "raw", args.signer)
crop_root = os.path.join("data", "crops", args.signer)
os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("data", "keypoints"), exist_ok=True)

keypoints, labels, stats = [], [], []

with vision.HandLandmarker.create_from_options(options) as landmarker:
    for letter in sorted(os.listdir(raw_dir)):
        letter_dir = os.path.join(raw_dir, letter)
        out_dir = os.path.join(crop_root, letter)
        os.makedirs(out_dir, exist_ok=True)
        kept = dropped = 0
        for name in sorted(os.listdir(letter_dir)):
            img = cv2.imread(os.path.join(letter_dir, name))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.hand_landmarks:
                dropped += 1
                continue

            #landmarks in pixel units
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in result.hand_landmarks[0]])

            #square crop around the hand, with a margin
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            half = max(x_max - x_min, y_max - y_min) / 2 * (1 + args.margin)
            x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
            y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
            if x1 - x0 < 10 or y1 - y0 < 10:
                dropped += 1
                continue
            crop = cv2.resize(img[y0:y1, x0:x1], (args.size, args.size))

            #keypoints: wrist at the origin, scaled by hand size
            #(distance from wrist to the base of the middle finger)
            rel = pts - pts[0]
            rel /= np.linalg.norm(rel[9, :2]) + 1e-6

            #mirror left hands so the model always sees one canonical hand
            if result.handedness[0][0].category_name == "Left":
                crop = cv2.flip(crop, 1)
                rel[:, 0] *= -1

            cv2.imwrite(os.path.join(out_dir, name), crop)
            keypoints.append(rel.flatten())
            labels.append(letter)
            kept += 1
        stats.append((letter, kept, dropped))
        print(f"{letter}: kept {kept}, dropped {dropped}")

np.savez(os.path.join("data", "keypoints", f"{args.signer}.npz"),
         keypoints=np.array(keypoints), labels=np.array(labels))
with open(os.path.join("results", f"crop_stats_{args.signer}.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["letter", "kept", "dropped"])
    writer.writerows(stats)