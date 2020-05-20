import numpy as np
import h5py
import cv2 as cv
from tqdm.auto import tqdm


class ORBModel:
    def __init__(self):
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def distance(self, im1, im2):
        if im1.shape[0] == 0 or im2.shape[0] == 0:
            return 1.0
        else:
            matches = self.bf.match(im1.astype(np.uint8), im2.astype(np.uint8))
            return 1.0-(len(matches)/self.orb.getMaxFeatures())

    def get_representations(self, save_file_path, dataset):
        try:
            f = h5py.File(save_file_path, "w")
            for im, image_identifier in tqdm(dataset):
                kp, des = self.orb.detectAndCompute(im, None)                
                f.create_dataset(image_identifier, data=des, dtype='f')
        finally:
            f.close()
