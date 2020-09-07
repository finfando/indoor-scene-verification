from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from indoorhack.tasks.utils import get_dataset, get_model, get_loader

dataset_type = "real_estate"
dataset_name = "sonar"
model_type = "orb"

main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
X = np.load(main_path / f"{dataset_name}_X.npy")
y = np.load(main_path / f"{dataset_name}_y.npy")

dataset_path = main_path / dataset_name
meta_path = main_path / f"{dataset_name}_meta.pkl"
repr_path = main_path / f"{dataset_name}_repr_{model_type}.hdf5"

# loader = lambda x: h5py.File(repr_path, "r")["/".join((tuple(x.parts[-2].split("_")) + (x.parts[-1].split(".")[0],)))][:]
loader = get_loader(repr_path=repr_path)
transform = None
meta = pd.read_pickle(meta_path)
dataset = get_dataset(
    dataset_type, path=dataset_path, meta=meta, loader=loader, transform=transform, use_label_encoding=True
)

def process_pairs(indices):
    print(' ', end='', flush=True) # hack for printing tqdm progress bar when using multiprocessing
    distances = []
    for i in tqdm(indices):
        im1_idx, im2_idx = X[i]
        im1 = dataset[im1_idx][0]
        im2 = dataset[im2_idx][0]
        distances.append(get_model(model_type).distance(im1, im2))
    return distances

distances = process_pairs(np.arange(len(X)))
# distances_flat = [number for n in results for number in n]
np.save(main_path / f"{dataset_name}_dist_{model_type}.npy", np.array(distances))
