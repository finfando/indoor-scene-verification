import torch
import torch.nn as nn
from torchvision.models import vgg16

from indoorhack.models.netvlad import NetVLAD

class IndoorNet(nn.Module):
    def __init__(self):
        super(IndoorNet, self).__init__()
        self.base = vgg16(pretrained=True)
        self.netvlad = NetVLAD(dim=512)
        
    def forward(self, x):
        x = self.base.features(x)
#         x = self.cnn.avgpool(x)
        x = self.netvlad(x)
        return x
