from pathlib import Path

import h5py
import numpy as np
from config.env import INDOORHACK_CHECKPOINT, NETVLAD_CHECKPOINT, TORCH_DEVICE
from indoorhack.datasets import RealEstateDataset, ScanDataset
from indoorhack.models import (FaceNetModel, HashModel, IndoorHackModel,
                               NetVLADModel, ORBModel)
from indoorhack.transforms import OpenCV2ImageFromPath
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_dataset(dataset_type, **kwargs):
    if dataset_type == "scan":
        return ScanDataset(**kwargs)
    elif dataset_type == "real_estate":
        return RealEstateDataset(**kwargs)
    else:
        raise NotImplementedError


def get_model(model_type, checkpoint=True):
    if model_type == "hash":
        return HashModel()
    elif model_type == "orb":
        return ORBModel()
    elif model_type == "netvlad":
        assert NETVLAD_CHECKPOINT is not None
        return NetVLADModel(device=TORCH_DEVICE, checkpoint=NETVLAD_CHECKPOINT)
    elif model_type == "facenet":
        return FaceNetModel(device=TORCH_DEVICE)
    elif model_type == "indoorhack":
        if checkpoint:
            assert INDOORHACK_CHECKPOINT is not None
            return IndoorHackModel(device=TORCH_DEVICE, checkpoint=INDOORHACK_CHECKPOINT)
        else:
            return IndoorHackModel(device=TORCH_DEVICE)
    elif model_type == "indoorhack-mobilenetv2":
        return IndoorHackModel(device=TORCH_DEVICE, base_model_architecture="mobilenetv2")
    else:
        raise NotImplementedError

def get_experiment(experiment_name):
    if experiment_name in ["hash", "orb", "netvlad", "facenet"]:
        return get_model(experiment_name)
    elif experiment_name == "indoorhack-v1":
        checkpoint_path = Path(__file__).resolve().parents[2] / "checkpoints" / "indoorhack_v1_4.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "indoorhack-v2":
        checkpoint_path = Path(__file__).resolve().parents[2] / "checkpoints" / "indoorhack_v2" / "2020-09-21_173122" / "indoorhack_v2_18.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "indoorhack-v21":
        checkpoint_path = Path(__file__).resolve().parents[2] / "checkpoints" / "indoorhack_v2" / "2020-09-29_114304" / "indoorhack_v2_checkpoint_58.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "indoorhack-v7-20":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "v6" / "indoorhack-v7-20.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)


    elif experiment_name == "innetvlad-v1-10":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v1" / "indoorhack-v7-10.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v1-20":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v1" / "indoorhack-v7-20.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v1-30":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v1" / "indoorhack-v7-30.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)

    elif experiment_name == "innetvlad-v2-10":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v2" / "indoorhack-v10-10.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v2-20":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v2" / "indoorhack-v10-20.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v2-30":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v2" / "indoorhack-v10-30.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)

    elif experiment_name == "innetvlad-v3-10":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v3" / "indoorhack-v12-10.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v3-20":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v3" / "indoorhack-v12-20.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    elif experiment_name == "innetvlad-v3-30":
        checkpoint_path = Path(__file__).resolve().parents[2] / "experiments" / "innetvlad-v3" / "indoorhack-v12-30.torch"
        return IndoorHackModel(device=TORCH_DEVICE, checkpoint=checkpoint_path)
    else:
        raise NotImplementedError

def get_loader(model_type=None, repr_path=None, dataset_type="real_estate"):
    if model_type == "orb":
        open_cv2 = OpenCV2ImageFromPath()
        return lambda x: open_cv2(str(x))
    elif repr_path and dataset_type == "real_estate":
        return lambda x: h5py.File(repr_path, "r")["/".join(x.parts[-3:])][:]
    elif repr_path and dataset_type == "scan":
        return lambda x: h5py.File(repr_path, "r")["/".join((tuple(x.parts[-2].split("_")) + (x.parts[-1].split(".")[0],)))][:]
    else:
        return None


def get_transformer(model_type):
    if model_type in ["netvlad", "facenet"] or model_type.startswith("indoorhack"):
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    else:
        return Compose([])


class EarlyStopping:
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics): # pylint: disable=method-hidden
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
