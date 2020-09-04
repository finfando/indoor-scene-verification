from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from indoorhack.datasets import ScanDataset, RealEstateDataset
from indoorhack.models import HashModel, ORBModel, NetVLADModel
from indoorhack.transforms import OpenCV2ImageFromPath
from confg.env import TORCH_DEVICE, NETVLAD_CHECKPOINT

def get_dataset(dataset_type, **kwargs):
    if dataset_type == "scan":
        raise NotImplementedError
    elif dataset_type == "real_estate":
        return RealEstateDataset(**kwargs)
    else:
        raise NotImplementedError


def get_model(model_type):
    if model_type == "hash":
        return HashModel()
    elif model_type == "orb":
        return ORBModel()
    elif model_type == "netvlad":
        return NetVLADModel(device=TORCH_DEVICE, checkpoint=NETVLAD_CHECKPOINT)
    else:
        raise NotImplementedError


def get_loader(model_type):
    if model_type == "orb":
        open_cv2 = OpenCV2ImageFromPath()
        return lambda x: open_cv2(str(x))
    else:
        return None


def get_transformer(model_type):
    if model_type == "netvlad":
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    else:
        return Compose([])
