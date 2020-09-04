from itertools import combinations, product, permutations
import numpy as np
import torch
from torch.nn.modules.distance import PairwiseDistance


def get_all_triplets(_, labels):
    triplets = []
    for l in np.unique(labels):
        mask = (l == labels)
        pos_indices = np.where(mask)[0]
        neg_indices = np.where(~mask)[0]

        # repeat pairs or not
#         pos_pairs = np.array(list(permutations(pos_indices, 2)))
        pos_pairs = np.array(list(combinations(pos_indices, 2)))
        
        for (a, p), d in zip(pos_pairs, pos_dists):
            neg_pairs = np.array(list(product([a], neg_indices)))
            triplets += [(a, p, n) for _, n in neg_pairs]
    return triplets


def get_triplets(embeddings, labels):
    pdist = PairwiseDistance(2)
    triplets = []
    pos_dist_out = []
    neg_dist_out = []
    for l in np.unique(labels):
        mask = (l == labels)
        pos_indices = np.where(mask)[0]
#         pos_pairs = np.array(list(permutations(pos_indices, 2)))
        pos_pairs = np.array(list(combinations(pos_indices, 2)))
        pos_dists = pdist(embeddings[pos_pairs[:,0]], embeddings[pos_pairs[:,1]])

        neg_indices = np.where(~mask)[0]
        
        for (a, p), d in zip(pos_pairs, pos_dists):
            neg_pairs = np.array(list(product([a], neg_indices)))
            neg_dists = pdist(embeddings[neg_pairs[:,0]], embeddings[neg_pairs[:,1]])
            _, idx_sorted = torch.sort(neg_dists)
            sorted_mask = np.isin(idx_sorted, np.where((neg_dists > d))[0])
            if sum(sorted_mask) == 0:
                continue
            idx = idx_sorted[sorted_mask][0]
            idx = idx_sorted[:][0]
            n = neg_pairs[idx][1]
            triplets.append((a, p, n))
            pos_dist_out.append(d)
            neg_dist_out.append(neg_dists[idx])
    return triplets, torch.stack(pos_dist_out), torch.stack(neg_dist_out)
