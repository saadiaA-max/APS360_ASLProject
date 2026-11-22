import csv
import os
import random
import cv2

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  #24 static letters (no J, Z)
PER_LETTER = 300
SIZE = 64

src_root = os.path.join("data", "asl_alphabet", "asl_alphabet_train")
nested = os.path.join(src_root, "asl_alphabet_train")  #the zip may nest the same name twice
if os.path.isdir(nested):
    src_root = nested

rng = random.Random(0)  #fixed seed so the sample is reproducible
stats = []
for letter in LETTERS:
    src_dir = os.path.join(src_root, letter)
    out_dir = os.path.join("data", "crops", "kaggle", letter)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(src_dir))
    chosen = rng.sample(files, min(PER_LETTER, len(files)))
    saved = 0
    for name in chosen:
        img = cv2.imread(os.path.join(src_dir, name))
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, name), img)
        saved += 1
    stats.append((letter, saved))
    print(f"{letter}: saved {saved}")

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "crop_stats_kaggle.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["letter", "saved"])
    writer.writerows(stats)