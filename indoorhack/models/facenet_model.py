import numpy as np
import h5py
import cv2 as cv
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from submodules.facenet_pytorch import MTCNN, InceptionResnetV1

class FaceNetModel:
    def __init__(self, device):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model.to(device)

    @staticmethod
    def distance(im1, im2):
        return np.linalg.norm(im1-im2)
    
#     def get_representations(self, save_file_path, dataset):
#         dataloader = DataLoader(
#             dataset, 
#             batch_size=64, 
#             shuffle=False, 
#             num_workers=5, 
#             pin_memory=True
#         )
#         try:
#             f = h5py.File(save_file_path, "w")
#             self.model.eval()
#             with torch.no_grad():
#                 for i, (tensor, image_identifier) in enumerate(tqdm(dataloader)):
#                     tensor = tensor.to(self.device)
#                     out = self.model(tensor)
#                     representation = out.detach().cpu().numpy()
#                     for i in range(len(image_identifier)):
#                         f.create_dataset(image_identifier[i], data=representation[i,:] , dtype='f')
#         finally:
#             f.close()
    def get_representations(self, save_file_path, dataset):
        dataloader = DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        try:
            f = h5py.File(save_file_path, "w")
            self.model.eval()
            with torch.no_grad():
                for i, (image, meta) in enumerate(tqdm(dataloader)):
                    image = image.to(self.device)
                    out = self.model(image)
                    representation = out.detach().cpu().numpy()
                    for i in range(len(meta[0])):
                        f.create_dataset("/".join([meta[0][i], meta[1][i], meta[2][i]]), data=representation[i,:], dtype='f')
        finally:
            f.close()