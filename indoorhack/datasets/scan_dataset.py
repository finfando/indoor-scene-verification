from typing import List
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ScanDataset(Dataset):
    def __init__(self, path: str, meta=None, loader=None, transform=None, use_label_encoding=False, return_seq=False):
        self.path = Path(path)
        self.transform = transform
        self.le = None
        self.return_seq = return_seq
        
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
            self.images = [meta2path(sce, sca, seq) for sce, sca, seq in tqdm(meta)]

        if use_label_encoding:
            scene_id, _, _ = zip(*self.meta)
            self.le = LabelEncoder()
            self.le.fit(scene_id)

    def __getitem__(self, idx):
        path, meta = self.images[idx], self.meta[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        item = (sample, )
        if self.le:
            item += (self.le.transform([meta[0]]), )
        else:
            item += (meta, )

        if self.return_seq:
            item += (np.array([int(meta[2])]), )
        return item

    def __len__(self):
        return len(self.images)
