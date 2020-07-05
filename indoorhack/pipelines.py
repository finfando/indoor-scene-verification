from torchvision.transforms import Compose, ToTensor, Normalize, Resize

def input_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def input_transform_small():
    return Compose([
        Resize((100, 100)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def pipeline():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
