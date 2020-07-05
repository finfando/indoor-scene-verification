import torch
import torch.nn as nn


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin, get_triplets_fn):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.get_triplets_fn = get_triplets_fn
        
    def forward(self, embedding, label):
        embedding = embedding.cpu()
        label = label.cpu().data.numpy()
        triplets, pos_dist, neg_dist = self.get_triplets_fn(embedding, label)
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss
