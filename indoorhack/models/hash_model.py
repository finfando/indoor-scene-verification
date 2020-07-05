import numpy as np
import h5py
import imagehash
from pathlib import Path
from tqdm.auto import tqdm


class HashModel:
    def __init__(self):
        self.func = imagehash.phash

    @staticmethod
    def distance(im1, im2):
        return np.bitwise_xor(im1, im2).sum()/64
    
    def get_representations(self, save_file_path, dataset):
        try:
            f = h5py.File(save_file_path, "w")
            for image, meta in tqdm(dataset, total=len(dataset)):
                im_hash = self.func(image)
                f.create_dataset("/".join(meta), data=im_hash.hash, dtype='bool')
        finally:
            f.close()