'''Script to launch experiments with hyperparameter search'''
import json
import argparse
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from dataset import get_loaders
from model import TransformerModel
from evaluate import ExperimentResult
from train import Trainer

def set_random_seeds(force_torch=False):
    '''Set all posible random seeds to make experiments reproducible'''
    random.seed(9000)
    np.random.seed(962)
    torch.manual_seed(2351)
    torch.cuda.manual_seed(2351)
    if force_torch:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

def run_experiment(config: dict):
    '''Run experiment from a given config'''

    output_path = Path(config['model_path']).parent
    if (output_path / 'learning_log.csv').exists():
        # Model already trained, skipping
        return

    set_random_seeds()

    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Get data
    loaders = get_loaders(config['feature_set'], config['split_folder'], config['split'],
                          config['batch_size'], config['permute'])
    train_loader, val_loader, test_loader = loaders
    nmodal = train_loader.dataset.nmodal
    nfeatures = train_loader.dataset.nfeatures

    # Define model
    model = TransformerModel(nmodal, 2, nfeatures, config['nhead'],
                             config['nhid'], config['nlayers'],
                             config['dropout']).to(device)

    # Train model
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'])
                                 #weight_decay=0.01)
    trainer = Trainer(model, loss_fn, optimizer, device)
    learning_log = trainer.main_loop(train_loader, val_loader,
                                     config['epochs'], config['criteria'],
                                     config['model_path'], config['stop_cutoff'])
    model.load_state_dict(torch.load(config['model_path']))

    # Get metrics
    val_result = ExperimentResult()
    val_result.compute(model, loss_fn, val_loader, device)
    test_result = ExperimentResult()
    test_result.compute(model, loss_fn, test_loader, device)

    write_outputs(val_result, test_result, learning_log, output_path)
    return model

def write_outputs(val_result, test_result, log, path):
    '''Write all the results in a given folder'''
    # Write predictions
    val_result.predictions.to_csv(path / 'val_predictions.csv', index=False)
    test_result.predictions.to_csv(path / 'test_predictions.csv', index=False)

    # Write scores
    with (path / 'val_scores.json').open('w', encoding='utf8') as fwrite:
        json.dump(val_result.scores, fwrite)
    with (path / 'test_scores.json').open('w', encoding='utf8') as fwrite:
        json.dump(test_result.scores, fwrite)

    # Write learning log in tidy version
    log = pd.DataFrame(log)
    log = log.melt(
        value_vars=['auroc', 'auprc', 'p@top{k}', 'enrichment@top{k}', 'loss'],
        id_vars=['epoch', 'subset'],
        var_name='metric',
        value_name='score'
    )
    log.to_csv(path / 'learning_log.csv', index=False)

def main():
    '''Parse arguments from command line'''
    parser = argparse.ArgumentParser(
        description='train the transformer model using params configured in a JSON file')
    parser.add_argument('config_path', type=str,
                        help='path where the config file is located')
    args = parser.parse_args()
    with open(args.config_path, encoding='utf8') as fread:
        config = json.load(fread)
    run_experiment(config)

if __name__ == "__main__":
    main()
