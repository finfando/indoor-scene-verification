from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from indoorhack.tasks.utils import get_dataset, get_model, get_loader


@click.command()
@click.option(
    "--dataset_type", type=click.Choice(["scan", "real_estate"]), required=True
)
@click.option(
    "--model_type",
    type=click.Choice(["hash", "orb", "netvlad", "facenet", "indoorhack-v6"]),
    required=True,
)
@click.option("--dataset_name", help="Name of dataset.", required=True)
def get_distances(dataset_type, model_type, dataset_name):

    main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
    X = np.load(main_path / f"{dataset_name}_X.npy")
    y = np.load(main_path / f"{dataset_name}_y.npy")

    dataset_path = main_path / dataset_name
    meta_path = main_path / f"{dataset_name}_meta.pkl"
    repr_path = main_path / f"{dataset_name}_repr_{model_type}.hdf5"

    loader = get_loader(repr_path=repr_path)
    transform = None
    meta = pd.read_pickle(meta_path)
    dataset = get_dataset(
        dataset_type,
        path=dataset_path,
        meta=meta,
        loader=loader,
        transform=transform,
        use_label_encoding=True,
    )

    distances = process_pairs(np.arange(len(X)), dataset, model_type, X)
    np.save(main_path / f"{dataset_name}_dist_{model_type}.npy", np.array(distances))


def process_pairs(indices, dataset, model_type, X):
    print(
        " ", end="", flush=True
    )  # hack for printing tqdm progress bar when using multiprocessing
    distances = []
    for i in tqdm(indices):
        im1_idx, im2_idx = X[i]
        im1 = dataset[im1_idx][0]
        im2 = dataset[im2_idx][0]
        distances.append(get_model(model_type).distance(im1, im2))
    return distances


if __name__ == "__main__":
    get_distances()
