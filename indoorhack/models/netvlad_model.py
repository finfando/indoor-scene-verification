import sys
import numpy as np
import h5py
import cv2 as cv
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from submodules.pytorch_NetVLAD.netvlad import NetVLAD


class NetVLADModel:
    def __init__(self, device, checkpoint):
        self.device = device
        checkpoint = torch.load(checkpoint, map_location=torch.device(self.device))
        
        pretrained = True
        encoder_dim = 512
        encoder = vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)
        self.model = nn.Module() 
        self.model.add_module('encoder', encoder)

        num_clusters = 64
        net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        self.model.add_module('pool', net_vlad)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

    @staticmethod
    def distance(im1, im2):
        return np.linalg.norm(im1-im2)
    
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
                    image_encoding = self.model.encoder(image.to(self.device))
                    vlad_encoding = self.model.pool(image_encoding)
                    representation = vlad_encoding.detach().cpu().numpy()
                    for i in range(len(meta[0])):
                        f.create_dataset("/".join([meta[0][i], meta[1][i], meta[2][i]]), data=representation[i,:], dtype='f')
        finally:
            f.close()
