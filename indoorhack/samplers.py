import numpy as np
from torch.utils.data import Sampler, BatchSampler
from tqdm.auto import tqdm

from indoorhack.utils import get_scene_indices, get_scene_scan_indices


class IdentitySampler(Sampler):
    def __init__(self, idx, size, total, stdev):
        self.idx = idx
        self.size = size
        self.total = total
        self.stdev = stdev

    def __iter__(self):
        sampled = []
        while len(sampled) < self.size:
            sampled_idx = np.round(np.random.normal(self.idx, self.stdev)).astype(int)
            if sampled_idx not in sampled and sampled_idx > 0 and sampled_idx <= self.total:
                sampled += [sampled_idx]
                yield sampled_idx

    def __len__(self):
        return len(self.size)
    
class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, n_batches, n_pos, stdev, n_scenes_in_batch, scene_indices_dict=None, scene_scan_indices_dict=None):
        self.dataset = dataset
        self.n_batches = n_batches
        self.n_pos = n_pos
        self.stdev = stdev
        self.n_scenes_in_batch = n_scenes_in_batch
        
        scene_id, scan_id, image_idx = zip(*self.dataset.meta)
        self.scene_id_arr, self.scan_id_arr, self.image_idx_arr = np.array(scene_id), np.array(scan_id), np.array(image_idx)
        self.uq_scene_id = np.unique(scene_id)
        self.uq_scene_scan_id = np.unique(list(zip(scene_id, scan_id)), axis=0)
        
        if scene_indices_dict:
            self.scene_indices_dict = scene_indices_dict
        else:
            self.scene_indices_dict = {}
            for s in tqdm(self.uq_scene_id, desc="scenes"):
                self.scene_indices_dict[s] = get_scene_indices(
                    s, 
                    self.scene_id_arr, 
                    self.scan_id_arr, 
                    self.image_idx_arr
                )
        
        if scene_scan_indices_dict:
            self.scene_scan_indices_dict = scene_scan_indices_dict
        else:
            self.scene_scan_indices_dict = {}
            for scene, scan in tqdm(self.uq_scene_scan_id, desc="scene scans"):
                self.scene_scan_indices_dict[(scene, scan)] = get_scene_scan_indices(
                    scene, 
                    scan, 
                    self.scene_id_arr, 
                    self.scan_id_arr, 
                    self.image_idx_arr
                )

    def __iter__(self):
        batch_count = 0
        while batch_count < self.n_batches:
            batch = []
            selected_scenes = np.random.choice(self.uq_scene_id, size=self.n_scenes_in_batch, replace=False)
            for s in selected_scenes:
                scene_indices = self.scene_indices_dict[s]
                selected_idx = np.random.choice(scene_indices, size=1, replace=False)[0]
                scene_id, scan_id, image_seq = self.dataset.meta[selected_idx]
                
                scene_scan_indices = self.scene_scan_indices_dict[(scene_id, scan_id)]
                i_sampler = IdentitySampler(int(image_seq), size=self.n_pos, total=scene_scan_indices.shape[0], stdev=self.stdev)
                batch += [scene_scan_indices[seq-1] for seq in i_sampler]
#                 batch += [(self.dataset[scene_scan_indices[seq-1]], self.dataset[scene_scan_indices[seq-1]][1][0]) for seq in i_sampler]
            batch_count += 1
            yield batch
    
    def __len__(self):
        return self.n_batches
