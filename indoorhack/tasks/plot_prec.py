from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from indoorhack.utils import scale_fix
from config.plot import EXPERIMENT_MAP

@click.command()
@click.option(
    "--dataset_type", type=click.Choice(["scan", "real_estate"]), required=True
)
@click.option("--dataset_name", help="Name of dataset.", required=True)
@click.option("--dataset_variant", help="Variant of a dataset.", multiple=True, required=False)
@click.option(
    "--experiment-name",
    type=click.Choice(["hash", "orb", "netvlad", "facenet", "indoorhack-v1", "indoorhack-v2"]),
    multiple=True,
    required=True,
)
def plot_prec(dataset_type, dataset_name, dataset_variant, experiment_name):
    main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
    plots_path = Path(__file__).resolve().parents[2] / "plots" / dataset_type
    if not dataset_variant:
        dataset_variant = [None, ]

    def get_data(dataset_variant):
        if dataset_variant:
            y = np.load(main_path / f"{dataset_name}_{dataset_variant}_y.npy")
        else:
            y = np.load(main_path / f"{dataset_name}_y.npy")

        distances = {}
        for e in experiment_name:
            if dataset_variant:
                X_dist_path = main_path / f"{dataset_name}_{dataset_variant}_dist_{e}.npy"
            else:
                X_dist_path = main_path / f"{dataset_name}_dist_{e}.npy"
            distances[e] = np.load(X_dist_path)

        return distances, y

    fig = plt.figure(figsize=(10 * len(dataset_variant), 10))
    fig.patch.set_facecolor('white')
    
    for i, v in enumerate(dataset_variant):
        distances, y = get_data(v)

        ax = plt.subplot(1, len(dataset_variant), i+1)
        for e, d in distances.items():
            name, color = EXPERIMENT_MAP[e]["name"], EXPERIMENT_MAP[e]["color"]
            distances_array = np.array(d)
            distances_array_scaled = scale_fix(distances_array, 0, 1)
            score = 1 - distances_array_scaled
            prec, rec, thresholds = precision_recall_curve(y, score)
            plt.plot(prec, rec, label=name, linewidth=2, c=color)

        plt.grid()
        plt.xlabel("Precision", size=25)
        plt.ylabel("Recall", size=25)
        plt.ylim((.8,1))
        plt.xlim((.8,1))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 30})
    plt.savefig(plots_path / f"prec_{dataset_name}_{'_'.join(experiment_name)}.pdf")
    plt.show()


if __name__ == "__main__":
    plot_prec()
