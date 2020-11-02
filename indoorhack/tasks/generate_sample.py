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
dataset_type = "scan"
transform = None
loader = None
meta = get_meta(dataset_type, "scannet_val")
scene_id, scan_id, image_idx = zip(*meta)
stdev = 20

dataset_val = get_dataset(
    dataset_type,
    path=SCAN_DATA_PATH / "scannet_val",
    meta=meta,
    loader=loader,
    transform=transform,
    use_label_encoding=True,
)

sampler = CustomBatchSampler(
    dataset_val,
    n_batches=100,
    n_pos=5,
    stdev=stdev,
    n_scenes_in_batch=15,
    scene_indices_dict=None,
    scene_scan_indices_dict=None,
)


selected_scenes_idx = np.random.choice(sampler.uq_scene_id.shape[0], size=sampler.n_scenes_in_batch, replace=False)
selected_scenes = sampler.uq_scene_id[selected_scenes_idx]

image_indices = []
for s in selected_scenes:
    scene_scan_mask = (sampler.uq_scene_scan_id[:,0] ==  s)
    scene_scans = sampler.uq_scene_scan_id[scene_scan_mask]

    scene_scan_chosen_idx = np.random.choice(scene_scans.shape[0], size=1, replace=False)
    scene_scan_chosen = scene_scans[scene_scan_chosen_idx][0]
    # print(scene_scan_chosen)
    
    scene_scan_chosen = sampler.scene_scan_indices_dict[tuple(scene_scan_chosen)]
#     print(scene_scan_chosen.shape)
    image_indices.append(scene_scan_chosen)
image_indices = np.hstack(image_indices)


meta_arr = np.array(meta)[image_indices]
meta_filtered = list(map(tuple, meta_arr))
pd.to_pickle(meta_filtered, main_path / "data" / "scan" / "representations_eval" / "meta_filtered.pkl")
