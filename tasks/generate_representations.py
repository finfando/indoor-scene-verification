import click
import pandas as pd
from indoorhack.datasets import ScanDataset, RealEstateDataset
from indoorhack.models import HashModel, ORBModel

from torchvision.transforms import Compose


@click.command()
@click.option(
    "--dataset_type",
    type=click.Choice(["scan", "real_estate"], case_sensitive=False),
    help="Type of dataset.",
)
@click.option("--dataset_path", help="Path to dataset files.")
@click.option("--save_meta_path", help="Path to save meta files.")
@click.option(
    "--model_type",
    type=click.Choice(["hash", "orb", "netvlad"], case_sensitive=False),
    help="Type of model.",
)
@click.option("--save_repr_path", help="Path to save repr files.")
def generate_representations(
    dataset_type, dataset_path, save_meta_path, model_type, save_repr_path
):
    loader = None
    transform = Compose([])
    dataset = get_dataset(
        dataset_type, path=dataset_path, loader=loader, transform=transform
    )
    pd.to_pickle(dataset.meta, save_meta_path)

    model = get_model(model_type)
    model.get_representations(save_repr_path, dataset)


def get_dataset(dataset_type, **kwargs):
    if dataset_type == "scan":
        raise NotImplementedError
    elif dataset_type == "real_estate":
        return RealEstateDataset(**kwargs)
    else:
        raise NotImplementedError


def get_model(model_type, **kwargs):
    if model_type == "hash":
        return HashModel(**kwargs)
    elif model_type == "orb":
        return ORBModel(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    generate_representations()
