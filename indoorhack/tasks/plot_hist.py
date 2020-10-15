from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from indoorhack.utils import scale_fix

dataset_type = "real_estate"
dataset_name = "sonar"
model_type = ["netvlad", "indoorhack-v2"]
main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
plots_path = Path(__file__).resolve().parents[2] / "plots" / dataset_type

y_path = main_path / f"{dataset_name}_y.npy"
y = np.load(y_path)

distances = {}
for m in model_type:
    X_dist_path = main_path / f"{dataset_name}_dist_{m}.npy"
    distances[m] = np.load(X_dist_path)

fig = plt.figure(figsize=(20,10))
fig.patch.set_facecolor('white')
for i, (name, d) in enumerate(distances.items()):
    distances_array = np.array(d)
    distances_array_scaled = scale_fix(distances_array, 0, 1)
    score = 1 - distances_array_scaled
    ax = plt.subplot(2, 2, i+1)
    plt.hist(score[y==1], bins=10, alpha=0.7, label="pos", color="lightgreen")
    plt.hist(score[y==0], bins=10, alpha=0.7, label="neg", color="pink")
    plt.grid()
    plt.ylim((0,700))
    plt.xlim((-0.05,1.05))
    plt.xticks([i/100 for i in range(0, 101, 10)])
    plt.title(name, size=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size': 30})
plt.savefig(plots_path / "hist.pdf")
plt.show()
