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

class BetaOnlineTripletLoss(nn.Module):
    def __init__(self, alpha, beta, get_triplets_fn):
        super(BetaOnlineTripletLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.get_triplets_fn = get_triplets_fn
        
    def forward(self, embedding, label):
        embedding = embedding.cpu()
        label = label.cpu().data.numpy()
        triplets, pos_dist, neg_dist = self.get_triplets_fn(embedding, label)
        hinge_dist = (
            torch.clamp(self.alpha - neg_dist, min=0.0) 
            + torch.clamp(pos_dist - self.beta, min=0.0)
        )
        loss = torch.mean(hinge_dist)
        return loss


class SeqLoss(nn.Module):
    def __init__(self, coeff, get_pairs_fn):
        super(SeqLoss, self).__init__()
        self.coeff = coeff
        self.get_pairs_fn = get_pairs_fn
        
    def forward(self, embedding, label, seq):
        embedding = embedding.cpu()
        label = label.cpu().data.numpy()
        pairs, dists, seqs = self.get_pairs_fn(embedding, label, seq)
        hinge_dist = torch.clamp(self.coeff * ((seqs - dists) ** 2), min=0.0)
        loss = torch.mean(hinge_dist)
        return loss
