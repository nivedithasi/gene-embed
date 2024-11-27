import torch
from torch import nn
from dataset import get_loaders
from model import TransformerModel
import json
import argparse
from pathlib import Path
import pandas as pd
from launcher import set_random_seeds


def load_model_from_config(config: dict, nmodal: int, nfeatures: int):
    """Load model from a given config"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Define model
    model = TransformerModel(
        nmodal,
        2,
        nfeatures,
        config["nhead"],
        config["nhid"],
        config["nlayers"],
        config["dropout"],
    ).to(device)

    return model


def train_random_model(model, dataloader, loss_fn, device):
    """Train a random model"""

    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        X, y = batch[0].to(device), batch[1].to(device)

        # Compute prediction and loss
        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        num_batches += 1

    total_loss /= num_batches
    return total_loss


def main():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(
        description="train the transformer model using params configured in a JSON file"
    )
    parser.add_argument(
        "config_path", type=str, help="path where the config file is located"
    )
    args = parser.parse_args()
    with open(args.config_path, encoding="utf8") as fread:
        config = json.load(fread)

    set_random_seeds()

    loaders = get_loaders(
        config["feature_set"],
        config["split_folder"],
        config["split"],
        config["batch_size"],
        config["permute"],
    )
    train_loader, val_loader, _ = loaders
    nmodal = train_loader.dataset.nmodal
    nfeatures = train_loader.dataset.nfeatures
    model = load_model_from_config(config, nmodal, nfeatures)
    loss_fn = nn.CrossEntropyLoss()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loss = train_random_model(model, train_loader, loss_fn, device)
    val_loss = train_random_model(model, val_loader, loss_fn, device)

    output_path = Path(config["model_path"]).parent
    log = pd.DataFrame.from_dict({"train_loss": [train_loss], "val_loss": [val_loss]})

    log.to_csv(output_path / "first_learning_log.csv", index=False)


if __name__ == "__main__":
    main()
