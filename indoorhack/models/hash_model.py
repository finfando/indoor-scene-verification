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
            for im, image_identifier in tqdm(dataset):
                im_hash = self.func(im)
                f.create_dataset(image_identifier, data=im_hash.hash, dtype='bool')
        finally:
            f.close()
