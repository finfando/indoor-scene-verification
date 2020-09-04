from pathlib import Path
import click
import pandas as pd
from torchvision.transforms import Compose
from indoorhack.tasks.utils import get_dataset, get_model, get_loader


@click.command()
@click.option("--dataset_type", type=click.Choice(["scan", "real_estate"]))
@click.option("--model_type", type=click.Choice(["hash", "orb", "netvlad"]))
@click.option("--dataset_name", help="Name of dataset.")
def generate_representations(dataset_type, model_type, dataset_name):
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
    transform = Compose([])
    meta = pd.read_pickle(meta_path)
    dataset = get_dataset(
        dataset_type, path=dataset_path, meta=meta, loader=loader, transform=transform
    )

    model = get_model(model_type)
    repr_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_repr_{model_type}.hdf5"
    )
    model.get_representations(repr_path, dataset)


if __name__ == "__main__":
    generate_representations()
