from typing import List
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class ScanNetIndoorImagePairDataset(Dataset):
    def __init__(self, path: str, transformer=None, step=None, stdev=None, n_scans=None, sample_neg=False):
        self.path = Path(path)
        self.transformer = None
        self.images = []
        self.images_pos = []
        self.images_neg = []
        self.step = step
        self.stdev = stdev
        
        scan_paths = [scan_path for scan_path in self.path.glob("**/color")]
        if n_scans:
            scan_paths = scan_paths[:n_scans]

        for scan_path in tqdm(scan_paths):
            scene_scan_id = scan_path.parts[-2]
            n_images = len([image_path for image_path in scan_path.glob("*.jpg")])
            
            if self.step and self.stdev:
                pos_pair_ids = []
                for i in range(n_images):
                    pos_pair_id = round(np.random.normal(i+self.step, self.stdev))
                    if pos_pair_id >= n_images:
                        break
                    pos_pair_ids.append(pos_pair_id)
    
                self.images_pos += [(scene_scan_id, str(i)+".jpg") for i in pos_pair_ids]
                self.images += [(scene_scan_id, str(i)+".jpg") for i in range(len(pos_pair_ids))]
            else:
                self.images += [(scene_scan_id, str(i)+".jpg") for i in range(n_images)]
    
        scene_scan_ids, img_ids = zip(*self.images)
        scene_ids, scan_ids = zip(*[i.split("_") for i in scene_scan_ids])
        self.images_meta = list(zip(scene_scan_ids, scene_ids, scan_ids, img_ids))
        self.scene_scan_ids = scene_scan_ids
        self.scene_ids = scene_ids

        if self.step and self.stdev and sample_neg:
            images_neg = []
            for scan_path in tqdm(scan_paths, desc="neg"):
                scene_scan_id = scan_path.parts[-2]
                n_images = (np.array(scene_scan_ids) == scene_scan_id).sum()
                scene_id, scan_id = scene_scan_id.split("_")
                indices = self.get_subset_indices(exclude_scenes=[scene_id])
                indices_subset = np.random.choice(indices, size=n_images, replace=True)
                images_neg += [(self[i][0][0], self[i][0][3]) for i in indices_subset]        
            self.images_neg = images_neg
        
        self.transformer = transformer

    def __getitem__(self, idx):
        meta = self.images_meta[idx]
        out = (meta, )
        
        if isinstance(idx, slice):
            image_path = [self.path / im_path / "color" / im_name for im_path, im_name in self.images[idx]]
            image_tr = [str(p) for p in image_path]
            if self.transformer:
                image_tr = self.transformer(image_tr)
            out = out + (image_tr, )
            
            if len(self.images_pos) > 0:
                image_path = [self.path / im_path / "color" / im_name for im_path, im_name in self.images_pos[idx]]
                image_pos_tr = [str(p) for p in image_path]
                if self.transformer:
                    image_pos_tr = self.transformer(image_pos_tr)
                out = out + (image_pos_tr, )
                
            if len(self.images_neg) > 0:
                image_path = [self.path / im_path / "color" / im_name for im_path, im_name in self.images_neg[idx]]
                image_neg_tr = [str(p) for p in image_path]
                if self.transformer:
                    image_neg_tr = self.transformer(image_neg_tr)
                out = out + (image_neg_tr, )
        
        elif isinstance(idx, int) or isinstance(idx, np.int64):
            image_path = self.path / self.images[idx][0] / "color" / self.images[idx][1]
            image_tr = str(image_path)
            if self.transformer:
                image_tr = self.transformer(image_tr)
            out = out + (image_tr, )
            
            if len(self.images_pos) > 0:
                image_path = self.path / self.images_pos[idx][0] / "color" / self.images_pos[idx][1]
                image_pos_tr = str(image_path)
                if self.transformer:
                    image_pos_tr = self.transformer(image_pos_tr)
                out = out + (image_pos_tr, )

            if len(self.images_neg) > 0:
                image_path = self.path / self.images_neg[idx][0] / "color" / self.images_neg[idx][1]
                image_neg_tr = str(image_path)
                if self.transformer:
                    image_neg_tr = self.transformer(image_neg_tr)
                out = out + (image_neg_tr, )
        else:
            raise NotImplementedError

        return out
    
    def __len__(self):
        return len(self.images)
    
    def get_subset_indices(self, exclude_scenes: List[str]):
        indices = np.where(~np.isin(np.array(self.scene_ids), exclude_scenes))[0]
        return indices.astype(int).tolist()
    
    def get_subset_scan_indices(self, scene_scan_id: List[str]):
        indices = np.where(np.isin(np.array(self.scene_scan_ids), scene_scan_id))[0]
        return indices.astype(int).tolist()