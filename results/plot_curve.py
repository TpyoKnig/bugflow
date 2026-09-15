"""Regenerate fly-learning-curve.png from the training CSV logs (run from repo root).

    .venv/bin/pip install matplotlib
    .venv/bin/python results/plot_curve.py
"""

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
for path, label, color in [("results/fly-training-log-synthetic.csv", "random stand-in network", "#888"),
                           ("results/fly-training-log-real.csv", "real MaleCNS connectome (313 neurons)", "#c33")]:
    with open(path) as fh:
        r = np.array([float(x["reward"]) for x in csv.DictReader(fh)])
    roll = np.convolve(r, np.ones(200) / 200, mode="valid")
    ax.plot(np.arange(len(roll)) + 200, roll, label=label, color=color, lw=1.8)
ax.axhline(1.0, ls=":", c="#ccc")
ax.set_xlabel("training episode")
ax.set_ylabel("reward (rolling mean, 200)")
ax.set_title("Fruit-fly brain learns to design n8n workflows\n(reward = correct trigger + action for the brief)")
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("results/fly-learning-curve.png", dpi=150)
print("saved results/fly-learning-curve.png")
