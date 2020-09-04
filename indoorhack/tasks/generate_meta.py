from pathlib import Path
import click
import pandas as pd
from torchvision.transforms import Compose
from indoorhack.tasks.utils import get_dataset


@click.command()
@click.option("--dataset_type", type=click.Choice(["scan", "real_estate"]))
@click.option("--dataset_name", help="Name of dataset.")
def generate_meta(dataset_type, dataset_name):
    loader = None
    transform = Compose([])
    dataset = get_dataset(
        dataset_type,
        path=Path(__file__).resolve().parents[2] / "data" / dataset_type / dataset_name,
        loader=loader,
        transform=transform,
    )
    pd.to_pickle(
        dataset.meta,
        Path(__file__).resolve().parents[2]
        / "data"
        / dataset_type
        / f"{dataset_name}_meta.pkl",
    )


if __name__ == "__main__":
    generate_meta()
