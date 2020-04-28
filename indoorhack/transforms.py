import pandas as pd
from PIL import Image
import cv2
import imagehash
from skimage import io, transform
from pathlib import Path
import h5py

class OpenPILImageFromPath:
    """Open path as PIL Image
    """
    def __call__(self, path: str) -> Image:
        return Image.open(path)

    
class OpenCV2ImageFromPath:
    """Open path as cv2 Image
    """
    def __call__(self, path: str):
        im = cv2.imread(path, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    
class GetHashFromPath:
    """Open path as PIL Image
    """
    def __init__(self, cache_path=None):
        if cache_path:
            self.cache = pd.read_pickle(cache_path)
        else:
            cache_path = {}
            
    def __call__(self, path: str) -> Image:
        scene = path.split("/")[-3]
        image_number = int(path.split("/")[-1].split(".")[-2])
        try:
            image_hash = self.cache[scene][image_number]
        except KeyError:
            image_hash = imagehash.phash(Image.open(path))
            try:
                self.cache[scene][image_number] = image_hash
            except KeyError:
                self.cache[scene] = {}
                self.cache[scene][image_number] = image_hash
        return image_hash

    
class GetRepr:
    """Open path as PIL Image
    """
    def __init__(self, path: Path):
        self.f = h5py.File(path, "r")

    def __call__(self, path: Path):
        return self.f[path.parts[-1]][:]
