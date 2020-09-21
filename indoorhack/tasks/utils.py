from pathlib import Path
import h5py
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from config.env import TORCH_DEVICE, NETVLAD_CHECKPOINT, INDOORHACK_V6_CHECKPOINT
from indoorhack.datasets import RealEstateDataset, ScanDataset
from indoorhack.models import HashModel, ORBModel, NetVLADModel, FaceNetModel, IndoorHackModel
from indoorhack.transforms import OpenCV2ImageFromPath


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
            assert INDOORHACK_V6_CHECKPOINT is not None
            return IndoorHackModel(device=TORCH_DEVICE, checkpoint=INDOORHACK_V6_CHECKPOINT)
        else:
            return IndoorHackModel(device=TORCH_DEVICE)
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
    if model_type in ["netvlad", "facenet", "indoorhack-v1"] or model_type.startswith("indoorhack"):
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    else:
        return Compose([])
