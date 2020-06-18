from typing import List
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ScanDataset(Dataset):
    def __init__(self, path: str, meta=None, loader=None, transform=None, target_transform=None):
        self.path = Path(path)
        self.transform = transform
        self.target_transform = target_transform
        if loader:
            self.loader = loader
        else:
            self.loader = default_loader
        path2meta = lambda x: (tuple(x[-2].split("_")) + (x[-1].split(".")[0],))
        meta2path = lambda sce, sca, seq: self.path / "_".join([sce, sca]) / (seq + ".jpg")
        if path and not meta:
            self.images = [i for i in tqdm(Path(self.path).glob("**/*.jpg"))]
            self.meta = [path2meta(i.parts) for i in self.images]
        elif meta:
            self.meta = meta
            self.images = [meta2path(sce, sca, seq) for sce, sca, seq in meta]

    def __getitem__(self, idx):
        path, meta = self.images[idx], self.meta[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, meta

    def __len__(self):
        return len(self.images)
