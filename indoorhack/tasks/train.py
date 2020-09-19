from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from tqdm.auto import tqdm

from config.env import SCAN_DATA_PATH, TORCH_DEVICE
from indoorhack.models import IndoorHackModel
from indoorhack.models.get_triplets import get_triplets
from indoorhack.models.loss import OnlineTripletLoss
from indoorhack.pipelines import pipeline
from indoorhack.samplers import CustomBatchSampler
from indoorhack.tasks.utils import get_dataset, get_model
from indoorhack.utils import generate_pos_neg_plot, scale_fix
from submodules.NetVLAD_pytorch.netvlad import EmbedNet, NetVLAD


def train(experiment_name, model_type, checkpoint, epochs):
    dataset_type = "scan"
    loader = None
    transform = pipeline()

    # train dataset
    dataloader_train = prepare_train_dataloader(loader, transform)

    # val dataset
    dataset_val = get_dataset(
        dataset_type,
        path=SCAN_DATA_PATH / "scannet_val",
        meta=get_meta(dataset_type, "scannet_val"),
        loader=loader,
        transform=transform,
        use_label_encoding=True,
    )
    validation_set = prepare_val_dataset()

    big_model = get_model(model_type, checkpoint=checkpoint)
    model = big_model.model
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = OnlineTripletLoss(margin=0.1, get_triplets_fn=get_triplets)
    pdist = PairwiseDistance(2)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (image, label) in enumerate(
            tqdm(dataloader_train, desc=str(epoch + 1) + " training"), 0
        ):
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
                print(
                    "(train)[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / interval)
                )
                running_loss = 0.0

        save_path = Path(__file__).resolve().parents[2] / "checkpoints"
        torch.save(
            model.state_dict(), 
            save_path / (experiment_name + "_" + str(epoch + 1) + ".torch")
        )

        model.eval()
        with torch.no_grad():
            distances = []
            labels = []
            for i, (pos, neg, label) in enumerate(
                tqdm(validation_set, desc=str(epoch + 1) + " validation"), 0
            ):
                pos_embedding = model(dataset_val[pos][0].unsqueeze(0).cuda())
                neg_embedding = model(dataset_val[neg][0].unsqueeze(0).cuda())
                distances.append(
                    pdist(pos_embedding, neg_embedding).cpu().data.numpy()[0]
                )
                labels.append(label)
            distances_sc = scale_fix(np.array(distances), 0, 1)
            auc_score = roc_auc_score(np.array(labels), 1 - distances_sc)
            print("(val)[%d] auc: %.3f" % (epoch + 1, auc_score))


def get_meta(dataset_type, dataset_name):
    meta_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_meta.pkl"
    )
    return pd.read_pickle(meta_path)


def prepare_val_dataset():
    path = Path(__file__).resolve().parents[2] / "data" / "scan" / "pairs-val.hdf5"
    f = h5py.File(path, "r")
    pairs_val = np.array(f["pairs-val"])
    f.close()
    true_mask = pairs_val[:, 2] == 1
    false_mask = pairs_val[:, 2] == 0

    true_sample = np.random.choice(
        np.arange(pairs_val[true_mask].shape[0]), size=1000, replace=False
    )
    false_sample = np.random.choice(
        np.arange(pairs_val[false_mask].shape[0]), size=9000, replace=False
    )

    validation_set = np.vstack(
        [pairs_val[true_mask][true_sample], pairs_val[false_mask][false_sample]]
    )

    return validation_set


def prepare_train_dataloader(loader, transform):
    dataset_train = get_dataset(
        "scan",
        path=SCAN_DATA_PATH / "scannet_train",
        meta=get_meta("scan", "scannet_train"),
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
        scene_scan_indices_dict=None,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_sampler=sampler, num_workers=8, pin_memory=True
    )
    return dataloader_train


if __name__ == "__main__":
    epochs = 1
    experiment_name = "indoorhack_v1"
    model_type = "indoorhack"
    checkpoint = False
    train(experiment_name, model_type, checkpoint, epochs)
