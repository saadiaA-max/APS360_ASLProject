import re
import sys
import matplotlib
matplotlib.use("Agg")  #save the figure without opening a window
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#usage: python src\plot_curve.py <log.txt> <output.png> "<title>"
log_path, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]

#PowerShell's Tee-Object often saves UTF-16, so check for a byte-order mark
raw = open(log_path, "rb").read()
text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="ignore")

pattern = re.compile(r"epoch\s+(\d+)\s+train loss ([\d.]+) acc ([\d.]+)\s+val loss ([\d.]+) acc ([\d.]+)")
rows = [tuple(float(v) for v in m.groups()) for m in pattern.finditer(text)]
epochs = [int(r[0]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs, [r[1] for r in rows], label="train")
axes[0].plot(epochs, [r[3] for r in rows], label="validation")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
axes[1].plot(epochs, [r[2] for r in rows], label="train")
axes[1].plot(epochs, [r[4] for r in rows], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
for ax in axes:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  #whole epochs on the x axis
fig.suptitle(title)
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"read {len(rows)} epochs, saved {out_path}")