from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from indoorhack.transforms import OpenPILImageFromPath

def input_transform():
    return Compose([
        OpenPILImageFromPath(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])