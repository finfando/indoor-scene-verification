from pathlib import Path
from itertools import combinations, product
import numpy as np
import pandas as pd

from indoorhack.tasks.utils import get_dataset, get_loader, get_transformer

dataset_type = "real_estate"
dataset_name = "sonar"
model_type = None

main_path = Path(__file__).resolve().parents[2] / "data" / dataset_type
dataset_path = main_path / dataset_name
meta_path = main_path / f"{dataset_name}_meta.pkl"

loader = get_loader(model_type)
transform = get_transformer(model_type)
meta = pd.read_pickle(meta_path)
dataset = get_dataset(
    dataset_type, path=dataset_path, meta=meta, loader=loader, transform=transform, use_label_encoding=True
)

apartment_id, scene_id, image_id = zip(*dataset.meta)
apartment_id_arr, scene_id_arr, image_idx_arr = np.array(apartment_id), np.array(scene_id), np.array(image_id)
uq_apartment_id = np.unique(apartment_id)
uq_apartment_scene_id = np.unique(list(zip(apartment_id, scene_id)), axis=0)

X_pos = []
X_neg = []
for a, s in uq_apartment_scene_id:
    pos_mask = np.logical_and(
        apartment_id_arr == a,
        scene_id_arr == s,
    )
    pos_indices = np.where(pos_mask)[0]
    pos_pairs = list(combinations(pos_indices, 2))
    X_pos += pos_pairs

    neg_mask = apartment_id_arr != a
    neg_indices = np.where(neg_mask)[0]
    neg_pairs = list(product(pos_indices, neg_indices))
    X_neg += neg_pairs

    print(a, s)
    print("Number of images in scene", sum(pos_mask))
    print("Number of negative images", sum(neg_mask))

X_pos = np.array(X_pos)
X_neg = np.array(X_neg)
X = np.vstack([X_pos, X_neg])
y = np.hstack([[1] * len(X_pos), [0] * len(X_neg)])

np.save(main_path / f"{dataset_name}_X.npy", X)
np.save(main_path / f"{dataset_name}_y.npy", y)
