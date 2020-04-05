import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class ScanNetIndoorDataset(Dataset):
    def __init__(self, path, transformer=None, step=40, stdev=5, seed=0, n_scenes=None):
        self.path = path
        self.transformer = transformer
        self.step = step
        self.stdev = stdev
        np.random.seed(seed)
        self.image_triplets = []
        self.scenes = [(a, len(c), ) for a, _, c in os.walk("/home/model/users/ff/scannet_data/scannet_val/images") if len(c) > 0]
        if n_scenes:
            self.scenes = self.scenes[:n_scenes]
        for scene in tqdm(self.scenes):
            self.image_triplets += self._get_image_triplets(scene)
            
    def __getitem__(self, idx):
        triplet = self.image_triplets[idx]
        if self.transformer:
            if isinstance(idx, slice):
                triplet = [tuple(self.transformer(i) for i in unit_triplet) for unit_triplet in triplet]
            else:
                triplet = tuple(self.transformer(i) for i in triplet)
        return triplet
    
    def __len__(self):
        return len(self.image_triplets)
    
    def _get_image_triplets(self, scene):    
        img_triplets = []
        for i in range(scene[1]):
            pos_pair = round(np.random.normal(i+self.step, self.stdev))
            if pos_pair >= scene[1]:
                break
            triplet = (
                os.path.join(scene[0], str(i)+".jpg"), 
                os.path.join(scene[0], str(pos_pair)+".jpg"),
                self._get_random_neg_pair(scene),
            )
            img_triplets.append(triplet)
        return img_triplets
        
    def _get_random_neg_pair(self, scene):
        other_scenes = self.scenes.copy()
        other_scenes.remove(scene)
        other_scene = other_scenes[np.random.randint(0, len(other_scenes))]
        pair_idx = np.random.randint(0, other_scene[1])
        return os.path.join(other_scene[0], str(pair_idx)+".jpg")
