from pathlib import Path
import click
import pandas as pd
from indoorhack.tasks.utils import get_dataset, get_model, get_loader, get_transformer, get_experiment


@click.command()
@click.option("--dataset_type", type=click.Choice(["scan", "real_estate"]), required=True)
@click.option("--model_type", type=click.Choice(["hash", "orb", "netvlad", "facenet", "indoorhack-v1"]), required=True)
@click.option("--dataset_name", help="Name of dataset.", required=True)
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
    transform = get_transformer(model_type)
    meta = pd.read_pickle(meta_path)
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
    generate_representations()
