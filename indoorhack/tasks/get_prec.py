from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

dataset_type = "real_estate"
dataset_name = "sonar"
model_type = ["hash", "orb", "facenet"]
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
    score = 1 - d
    prec, rec, thresholds = precision_recall_curve(y, score)
    plt.plot(prec, rec, label=name)

plt.grid()
plt.xlabel("Precision", size=25)
plt.ylabel("Recall", size=25)
plt.ylim((-0.05,1.05))
plt.xlim((-0.05,1.05))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={'size': 30})
plt.title("Real estate dataset")
plt.savefig(plots_path / "prec.pdf")
plt.show()
