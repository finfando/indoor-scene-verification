from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from indoorhack.tasks.utils import (get_dataset, get_experiment, get_loader,
                                    get_model)

NTHREADS = 8
dataset, model_type_global, X = None, None, None

@click.command()
@click.option(
    "--dataset_type", type=click.Choice(["scan", "real_estate"]), required=True
)
@click.option(
    "--model_type",
    type=click.Choice(["hash", "orb", "netvlad", "facenet", "indoorhack-v1", "indoorhack-v2"]),
    required=True,
)
@click.option("--dataset_name", help="Name of dataset.", required=True)
@click.option("--dataset_variant", help="Variant of a dataset.", required=False)
def get_distances(dataset_type, model_type, dataset_name, dataset_variant):
    global dataset, model_type_global, X
    model_type_global = model_type
    main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
    if dataset_variant:
        X = np.load(main_path / f"{dataset_name}_{dataset_variant}_X.npy")
        y = np.load(main_path / f"{dataset_name}_{dataset_variant}_y.npy")
    else:
        X = np.load(main_path / f"{dataset_name}_X.npy")
        y = np.load(main_path / f"{dataset_name}_y.npy")

    dataset_path = main_path / dataset_name
    meta_path = main_path / f"{dataset_name}_meta.pkl"
    repr_path = main_path / f"{dataset_name}_repr_{model_type}.hdf5"

    loader = get_loader(repr_path=repr_path, dataset_type=dataset_type)
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
    
    ids = np.arange(len(X))
    if NTHREADS == 1:
        distances_flat = process_pairs(ids, )
    elif NTHREADS > 1:
        splits = np.array_split(ids, NTHREADS)
        with Pool(NTHREADS) as pool:
            results = pool.map(process_pairs, splits)
        distances_flat = [number for n in results for number in n]
        # distances_arr = np.array(distances_flat)
        # distances_arr_scaled = scale_fix(distances_arr, 0, 1)
    np.save(main_path / f"{dataset_name}_dist_{model_type}.npy", np.array(distances_flat))


def process_pairs(indices):
    global dataset, model_type_global, X
    print(
        " ", end="", flush=True
    )  # hack for printing tqdm progress bar when using multiprocessing
    distances = []
    for i in tqdm(indices):
        im1_idx, im2_idx = X[i]
        im1 = dataset[im1_idx][0]
        im2 = dataset[im2_idx][0]
        distances.append(get_experiment(model_type_global).distance(im1, im2))
    return distances


if __name__ == "__main__":
    get_distances()
