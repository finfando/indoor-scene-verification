from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from tqdm.auto import tqdm

from config.env import TORCH_DEVICE
from indoorhack.models.get_triplets import get_triplets
from indoorhack.models.loss import OnlineTripletLoss
from indoorhack.pipelines import pipeline
from indoorhack.samplers import CustomBatchSampler
from indoorhack.tasks.utils import get_dataset
from indoorhack.utils import generate_pos_neg_plot, scale_fix
from submodules.NetVLAD_pytorch.netvlad import NetVLAD, EmbedNet

dataset_type = "scan"
dataset_name = "scannet_train"

dataset_path = (
        Path(__file__).resolve().parents[2] / "data" / dataset_type / dataset_name
)
meta_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_meta.pkl"
)

meta = pd.read_pickle(meta_path)
loader = None
transform = pipeline()

dataset_train = get_dataset(
    dataset_type,
    path=dataset_path,
    meta=meta,
    loader=loader,
    transform=transform,
    use_label_encoding=True,
)

sampler = CustomBatchSampler(
    dataset_train,
    n_batches=100,
    n_pos=5,
    stdev=20,
    n_scenes_in_batch=15,
    scene_indices_dict=None,
    scene_scan_indices_dict=None
)

dataloader_train = DataLoader(
    dataset_train,
    batch_sampler=sampler,
    num_workers=4,
    pin_memory=True
)

base_model = vgg16(pretrained=False).features

dim = list(base_model.parameters())[-1].shape[0]
net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)

model = EmbedNet(base_model, net_vlad).to(TORCH_DEVICE)

# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = OnlineTripletLoss(margin=0.1, get_triplets_fn=get_triplets)


EPOCHS = 100
pdist = PairwiseDistance(2)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (image, label) in enumerate(tqdm(dataloader_train, desc=str(epoch + 1) + " training"), 0):
        embeddings = model(image.cuda())
        try:
            loss = criterion(embeddings, label)
        except RuntimeError as e:
            print(e)
            continue
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        interval = 10
        if (i + 1) % interval == 0:
            print('(train)[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / interval))
            running_loss = 0.0

    torch.save(model.state_dict(), "models/indoorhack_v6_" + str(epoch + 1) + ".torch")

    model.eval()
    with torch.no_grad():

        pd_lst = []
        nd_lst = []
        for i, (a, p, n) in enumerate(tqdm(triplets_val[:], desc=str(epoch + 1) + " validation"), 0):
            ae = model(dataset_val[a][0].unsqueeze(0).cuda())
            pe = model(dataset_val[p][0].unsqueeze(0).cuda())
            ne = model(dataset_val[n][0].unsqueeze(0).cuda())

            pd = pdist(ae, pe)
            nd = pdist(ae, ne)
            pd_lst.append(pd.cpu().data.numpy()[0])
            nd_lst.append(nd.cpu().data.numpy()[0])
        generate_pos_neg_plot(pd_lst, nd_lst, epoch + 1, Path("experiments/v6"))

        distances = []
        labels = []
        for i, (pos, neg, label) in enumerate(tqdm(validation_set, desc=str(epoch + 1) + " validation"), 0):
            pos_embedding = model(dataset_val[pos][0].unsqueeze(0).cuda())
            neg_embedding = model(dataset_val[neg][0].unsqueeze(0).cuda())
            distances.append(pdist(pos_embedding, neg_embedding).cpu().data.numpy()[0])
            labels.append(label)
        distances_sc = scale_fix(np.array(distances), 0, 1)
        auc_score = roc_auc_score(np.array(labels), 1 - distances_sc)
        print('(val)[%d] auc: %.3f' %
              (epoch + 1, auc_score))