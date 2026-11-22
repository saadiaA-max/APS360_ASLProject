import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  #save figures without opening a window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs="+", default=["saadia"])
parser.add_argument("--val", nargs="+", default=["p1"])
parser.add_argument("--test", nargs="+", default=None)  #leave unset until the final report
args = parser.parse_args()


def load(signers):
    xs, ys = [], []
    for name in signers:
        data = np.load(os.path.join("data", "keypoints", f"{name}.npz"))
        xs.append(data["keypoints"])
        ys.append(data["labels"])
    return np.concatenate(xs), np.concatenate(ys)


def report(split_name, signers, clf, letters, save_as):
    X, y = load(signers)
    pred = clf.predict(X)
    print(f"\n{split_name} ({', '.join(signers)}): {len(y)} samples, "
          f"accuracy = {accuracy_score(y, pred):.3f}")

    cm = confusion_matrix(y, pred, labels=letters)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("per-letter accuracy:",
          ", ".join(f"{l}:{a:.2f}" for l, a in zip(letters, per_class)))

    #most frequent mistakes (off-diagonal cells of the confusion matrix)
    pairs = [(cm[i, j], letters[i], letters[j])
             for i in range(len(letters)) for j in range(len(letters))
             if i != j and cm[i, j] > 0]
    print("most confused (true -> predicted):")
    for count, true_l, pred_l in sorted(pairs, reverse=True)[:5]:
        print(f"  {true_l} -> {pred_l}: {count}")

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(letters)), letters)
    plt.yticks(range(len(letters)), letters)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Random forest on keypoints: {split_name}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_as, dpi=150)
    plt.close()


X_train, y_train = load(args.train)
letters = sorted(set(y_train))
print(f"training on {len(y_train)} samples from {args.train}, {len(letters)} letters")

clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)

os.makedirs("results", exist_ok=True)
report("validation", args.val, clf, letters, os.path.join("results", "baseline_confusion_val.png"))
if args.test:
    report("test", args.test, clf, letters, os.path.join("results", "baseline_confusion_test.png"))