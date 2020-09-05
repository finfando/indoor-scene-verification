from typing import List
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class RealEstateDataset(Dataset):
    def __init__(self, path: str, meta=None, loader=None, transform=None,
                 use_label_encoding=False):
        self.path = Path(path)
        self.transform = transform
        self.le = None

        if loader:
            self.loader = loader
        else:
            self.loader = default_loader

        path2meta = lambda x: (x[-3], x[-2], x[-1])
        meta2path = lambda apartment_id, scene_id, image_id: self.path / apartment_id / scene_id / image_id

        if path and not meta:
            self.images = [i for i in tqdm(Path(self.path).glob("**/*.jpg"))]
            self.meta = [path2meta(i.parts) for i in self.images]
        elif meta:
            self.meta = meta
            self.images = [meta2path(a, s, i) for a, s, i in tqdm(meta)]

        if use_label_encoding:
            apartment_id, scene_id, _ = zip(*self.meta)
            self.le = LabelEncoder()
            self.le.fit(["_".join([a, s]) for a, s in list(zip(apartment_id, scene_id))])

    def __getitem__(self, idx):
        path, meta = self.images[idx], self.meta[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.le:
            return sample, self.le.transform([
                "_".join([meta[0], meta[1]])
            ])
        else:
            return sample, meta

    def __len__(self):
        return len(self.images)
