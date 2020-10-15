import numpy as np
import h5py
import cv2 as cv
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from submodules.NetVLAD_pytorch.netvlad import NetVLAD, EmbedNet, TripletNet
from config.env import NTHREADS

dataset = None

class IndoorHackModel:
    def __init__(self, device, checkpoint=None):
        self.device = device
        base_model = vgg16(pretrained=True).features
        dim = list(base_model.parameters())[-1].shape[0]
        net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
        self.model = EmbedNet(base_model, net_vlad).to(device)

        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location=torch.device(self.device)) # pylint: disable=no-member
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.to(device)

    @staticmethod
    def distance(im1, im2):
        return np.linalg.norm(im1-im2)
    
    def get_representations(self, save_file_path, dataset):
        dataloader = DataLoader(
            dataset, 
            batch_size=64,
            shuffle=False, 
            num_workers=NTHREADS,
            pin_memory=False
        )
        try:
            f = h5py.File(save_file_path, "w")
            self.model.eval()
            with torch.no_grad():
                for i, (tensor, meta) in enumerate(tqdm(dataloader)):
                    tensor = tensor.to(self.device)
                    out = self.model(tensor)
                    representation = out.detach().cpu().numpy()
                    for i in range(len(meta[0])):
                        f.create_dataset("/".join([meta[0][i], meta[1][i], meta[2][i]]), data=representation[i,:] , dtype='f')
        finally:
            f.close()
