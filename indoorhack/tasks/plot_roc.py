from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from indoorhack.utils import scale_fix

dataset_type = "real_estate"
dataset_name = "sonar"
model_type = ["hash", "orb", "netvlad", "facenet"]
main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
plots_path = Path(__file__).resolve().parents[2] / "plots" / dataset_type

y_path = main_path / f"{dataset_name}_y.npy"
y = np.load(y_path)

distances = {}
for m in model_type:
    X_dist_path = main_path / f"{dataset_name}_dist_{m}.npy"
    distances[m] = np.load(X_dist_path)

fig = plt.figure(figsize=(10,10))
fig.patch.set_facecolor('white')
for name, d in distances.items():
    distances_array = np.array(d)
    distances_array_scaled = scale_fix(distances_array, 0, 1)
    score = 1 - distances_array_scaled
    fpr, tpr, thresholds = roc_curve(y, score)
    plt.plot(fpr, tpr, label=name, linewidth=2)

plt.plot([0, 1], [0, 1], c="blue", linewidth=0.5, linestyle="--")
plt.grid()
plt.xlabel("Fall-out", size=25)
plt.ylabel("Recall", size=25)
plt.ylim((-0.05,1.05))
plt.xlim((-0.05,1.05))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={'size': 30})
plt.title("Real estate dataset", size=30)
plt.savefig(plots_path / "roc.pdf")
plt.show()
