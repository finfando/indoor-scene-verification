from indoorhack.datasets import ScanDataset, RealEstateDataset
from indoorhack.models import HashModel, ORBModel
from indoorhack.transforms import OpenCV2ImageFromPath


def get_dataset(dataset_type, **kwargs):
    if dataset_type == "scan":
        raise NotImplementedError
    elif dataset_type == "real_estate":
        return RealEstateDataset(**kwargs)
    else:
        raise NotImplementedError


def get_model(model_type, **kwargs):
    if model_type == "hash":
        return HashModel(**kwargs)
    elif model_type == "orb":
        return ORBModel(**kwargs)
    else:
        raise NotImplementedError


def get_loader(model_type):
    if model_type == "orb":
        open_cv2 = OpenCV2ImageFromPath()
        return lambda x: open_cv2(str(x))
    else:
        return None
