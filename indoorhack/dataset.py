import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset


class ScanNetIndoorTripletDataset(Dataset):
    def __init__(self, path, transformer=None, step=40, stdev=5, seed=0, n_scenes=None):
        self.path = path
        self.transformer = transformer
        self.step = step
        self.stdev = stdev
        np.random.seed(seed)
        self.image_triplets = []
        self.scenes = [(a, len(c), ) for a, _, c in os.walk(self.path) if len(c) > 0]
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
        other_scenes = self.scenes.copy()
        for s in self.scenes:
            if s[0].split("/")[-2].startswith(scene[0].split("/")[-2][:-2]):
                other_scenes.remove(s)

        img_triplets = []
        for i in range(scene[1]):
            pos_pair = round(np.random.normal(i+self.step, self.stdev))
            if pos_pair >= scene[1]:
                break
            triplet = (
                os.path.join(scene[0], str(i)+".jpg"), 
                os.path.join(scene[0], str(pos_pair)+".jpg"),
                self._get_random_neg_pair(other_scenes),
            )
            img_triplets.append(triplet)
        return img_triplets
        
    def _get_random_neg_pair(self, other_scenes):
        other_scene = other_scenes[np.random.randint(0, len(other_scenes))]
        pair_idx = np.random.randint(0, other_scene[1])
        return os.path.join(other_scene[0], str(pair_idx)+".jpg")

class ScanNetIndoorImageDataset(Dataset):
    def __init__(self, path, transformer=None, n_scenes=None):
        self.path = path
        self.transformer = transformer
        self.images = []
        self.scenes = [(a, len(c), ) for a, _, c in os.walk(self.path) if len(c) > 0]
        if n_scenes:
            self.scenes = self.scenes[:n_scenes]
        for scene in tqdm(self.scenes):
            self.images += self._get_images(scene)
            
    def __getitem__(self, idx):
        image = self.images[idx]
        path = image
        if self.transformer:
            if isinstance(idx, slice):
                image = [self.transformer(unit_image) for unit_image in image]
            else:
                image = self.transformer(image)
        return image, path
    
    def __len__(self):
        return len(self.images)
    
    def _get_images(self, scene):    
        img = []
        for i in range(scene[1]):
            img.append(os.path.join(scene[0], str(i)+".jpg"))
        return img

class RealEstateListingTripletDataset(Dataset):
    def __init__(self, path, transformer=None, seed=0, n_scenes=None):
        self.path = path
        self.transformer = transformer
        np.random.seed(seed)
        self.image_triplets = []
        self.scenes = [(Path(a), c) for a, _, c in os.walk(self.path) if len(c) > 0]
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
        other_scenes = self.scenes.copy()
        other_scenes.remove(scene)

        img_triplets = []
        
        
        for i1, i2 in itertools.combinations(scene[1], 2):
            triplet = (
                scene[0] / i1, 
                scene[0] / i2,
                self._get_random_neg_pair(other_scenes),
            )
            img_triplets.append(triplet)
        return img_triplets

    def _get_random_neg_pair(self, other_scenes):
        other_scene = other_scenes[np.random.randint(0, len(other_scenes))]
        pair_idx = np.random.choice(other_scene[1], 1)[0]
        return other_scene[0] / pair_idx

class RealEstateListingImageDataset(Dataset):
    def __init__(self, path, transformer=None, n_scenes=None):
        self.path = path
        self.transformer = transformer
        self.images = []
        self.scenes = [(a, c) for a, _, c in os.walk(self.path) if len(c) > 0]
        if n_scenes:
            self.scenes = self.scenes[:n_scenes]
        for scene in tqdm(self.scenes):
            self.images += self._get_images(scene)
            
    def __getitem__(self, idx):
        image = self.images[idx]
        path = image
        if self.transformer:
            if isinstance(idx, slice):
                image = [self.transformer(unit_image) for unit_image in image]
            else:
                image = self.transformer(image)
        return image, path
    
    def __len__(self):
        return len(self.images)
    
    def _get_images(self, scene):    
        img = []
        for i in scene[1]:
            img.append(os.path.join(scene[0], i))
        return img
