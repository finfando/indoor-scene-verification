from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

from indoorhack.utils import scale_fix
from indoorhack.tasks.train import get_meta
from indoorhack.pipelines import pipeline
from indoorhack.samplers import CustomBatchSampler
from indoorhack.tasks.utils import get_dataset, get_model, get_loader, get_transformer, get_experiment
from config.env import SCAN_DATA_PATH


main_path = Path(__file__).resolve().parents[2]

meta_filtered = pd.read_pickle(main_path / "data" / "scan" / "representations_eval" / "meta_filtered.pkl")
dataset_type = "scan"
transform = pipeline()
loader = None

dataset_val_filtered = get_dataset(
    dataset_type,
    path=SCAN_DATA_PATH / "scannet_val",
    meta=meta_filtered,
    loader=loader,
    transform=transform,
    use_label_encoding=False,
)


def get_repr(experiment_name):
    model = get_experiment(experiment_name)
    repr_path = (
        main_path
        / "data"
        / "scan"
        / "representations_eval"
        / f"scannet_val_repr_{experiment_name}.hdf5"
    )
    model.get_representations(repr_path, dataset_val_filtered)


experiment_names = [
    # "netvlad", 
    # "innetvlad-v1-10",
    # "innetvlad-v1-20",
    # "innetvlad-v1-30",
    # "innetvlad-v2-10",
    # "innetvlad-v2-20",
    # "innetvlad-v2-30",
    "innetvlad-v3-10",
    "innetvlad-v3-20",
    "innetvlad-v3-30",
]
for en in experiment_names:
    get_repr(en)
