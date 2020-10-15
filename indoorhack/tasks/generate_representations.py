from pathlib import Path
import click
import numpy as np
import pandas as pd
from indoorhack.tasks.utils import get_dataset, get_model, get_loader, get_transformer, get_experiment
from config.env import SCAN_DATA_PATH

@click.command()
@click.option("--dataset_type", type=click.Choice(["scan", "real_estate"]), required=True)
@click.option("--model_type", type=click.Choice(["hash", "orb", "netvlad", "facenet", "indoorhack-v1", "indoorhack-v2", "indoorhack-v21"]), required=True)
@click.option("--dataset_name", help="Name of dataset.", required=True)
@click.option("--val_only", help="Generate only representations of images in val dataset", is_flag=True)
def generate_representations(dataset_type, model_type, dataset_name, val_only):
    if dataset_type == "scan":
        dataset_path = SCAN_DATA_PATH / dataset_name
    else:
        dataset_path = (
            Path(__file__).resolve().parents[2] / "data" / dataset_type / dataset_name
        )
    meta_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_meta.pkl"
    )
    loader = get_loader(model_type)
    transform = get_transformer(model_type)
    meta = pd.read_pickle(meta_path)
    if val_only and dataset_type == "scan":
        path10 = Path("/home/filip/indoor_scene_verification/data/scan/scannet_val_10_X.npy")
        path20 = Path("/home/filip/indoor_scene_verification/data/scan/scannet_val_20_X.npy")
        path30 = Path("/home/filip/indoor_scene_verification/data/scan/scannet_val_30_X.npy")
        data10 = np.load(path10)
        data20 = np.load(path20)
        data30 = np.load(path30)
        data = np.vstack([data10, data20, data30]).flatten()
        uq_indices = np.unique(data)
        meta_arr = np.array(meta)[uq_indices]
        meta = list(map(tuple, meta_arr))
    dataset = get_dataset(
        dataset_type, path=dataset_path, meta=meta, loader=loader, transform=transform
    )

    model = get_experiment(model_type)
    repr_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_repr_{model_type}.hdf5"
    )
    model.get_representations(repr_path, dataset)


if __name__ == "__main__":
    generate_representations() # pylint: disable=no-value-for-parameter
