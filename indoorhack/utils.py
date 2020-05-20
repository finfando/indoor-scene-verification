from torchvision.transforms import Compose
from indoorhack.transforms import OpenPILImageFromPath, GetRepr
from indoorhack.datasets import RealEstateListingTripletDataset, ScanNetIndoorTripletDataset

def get_dataset(data_path, representation_path)
    TRANSFORMER = Compose([
        GetRepr(representation_path),
    ])
    
    dataset = RealEstateListingTripletDataset(data_path, TRANSFORMER)
    return dataset
