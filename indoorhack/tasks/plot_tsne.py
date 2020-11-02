from pathlib import Path

import click
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from indoorhack.utils import scale_fix
from indoorhack.tasks.train import get_meta
from indoorhack.pipelines import pipeline
from indoorhack.samplers import CustomBatchSampler
from indoorhack.tasks.utils import get_dataset, get_model, get_loader, get_transformer, get_experiment
from config.env import SCAN_DATA_PATH, NTHREADS

main_path = Path(__file__).resolve().parents[2]


@click.command()
@click.option(
    "-e",
    "--experiment_name", 
    required=True,
)
def plot_tsne(experiment_name):
    repr_path = (
        main_path
        / "data"
        / "scan"
        / "representations_eval"
        / f"scannet_val_repr_{experiment_name}.hdf5"
    )
    dataset_type = "scan"
    loader = get_loader(repr_path=repr_path, dataset_type=dataset_type)
    transform = None
    meta_filtered = pd.read_pickle(main_path / "data" / "scan" / "representations_eval" / "meta_filtered.pkl")
    dataset = get_dataset(
        dataset_type,
        path=SCAN_DATA_PATH / "scannet_val",
        meta=meta_filtered,
        loader=loader,
        transform=transform,
        use_label_encoding=False,
    )
    Xx = []
    yy = []
    for emb, lab in tqdm(dataset):
        Xx.append(emb)
        yy.append(lab)
    scene_id, scan_id, image_idx = zip(*meta_filtered)
    image_idx = np.array(image_idx).astype(int)
    mask = np.logical_and(
        ((image_idx % 10) == 0),
        (image_idx < 800)
    )
    X = np.stack(Xx)[mask]
    y = np.array(yy)[mask]

    X_embedded = TSNE(n_components=2, n_jobs=NTHREADS).fit_transform(X)
    X_embedded.shape

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax = plt.subplot(1, 1, 1)
    for i in np.unique(y[:,0])[:10]:
        plt.scatter(X_embedded[y[:,0]==i,0], X_embedded[y[:,0]==i,1], label=i)
        
        # for (j, k), imidx in zip(X_embedded[y[:,0]==i,:][::10], y[y[:,0]==i,2][::10]):
        #     plt.annotate(imidx, (j, k), size=9)
    # plt.legend(prop={'size': 15})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(main_path / "plots" / dataset_type / f"tsne_{experiment_name}.pdf")

if __name__ == "__main__":
    plot_tsne() # pylint: disable=no-value-for-parameter
