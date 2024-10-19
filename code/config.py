'''Manage hyperparams of the experiment'''
import json
import argparse
from pathlib import Path
import hashlib
import numpy as np

from random_config import get_head_opts, hash_conf, write_config


def parse_configs(split_folder, split, feature_set, nhead, nhid, nlayers, learning_rate, dropout, permute, output_dir, seed=42):
    '''
    Generate random hyperparams configurations
    '''
    assert nhead in get_head_opts(feature_set)
    rng = np.random.default_rng(seed)

    init_range = float(rng.uniform(10. ** -3, 10. ** -1.))
    config = {'split': split,
            'split_folder': split_folder,
              'feature_set': feature_set,
                'batch_size': 1024,
                'nhead': nhead,
                'nhid': nhid,
                'nlayers': nlayers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'epochs': 200,
                'criteria': 'auprc',
                'permute': permute,
                'init_range': init_range,
                'stop_cutoff': 200}
    
    hash_id = hash_conf(config)
    config['hash_id'] = hash_id
    write_config(config, output_dir)

def main():
    '''Parse arguments from command line'''
    parser = argparse.ArgumentParser(
        description='generate config files with different hyperparams')
    parser.add_argument('split_folder', type=str)
    parser.add_argument('split', type=str,
                        choices=['compound', 'gene', 'pair'],
                        help='experimental split')
    parser.add_argument('feature_set', type=str,
                        choices=['crispr', 'crispr_orf', 'orf'],
                        help='features used to train')
    parser.add_argument('nhead', type=int,
                        help='number of attention heads')
    parser.add_argument('nhid', type=int,
                        help='size of hidden dimension')
    parser.add_argument('nlayers', type=int,
                        help='number of layers')
    parser.add_argument('learning_rate', type=float,
                        help='learning rate')
    parser.add_argument('dropout', type=float,
                        help='dropout rates')
    parser.add_argument('output_dir',
                        help='path to store the logs and predictions')
    parser.add_argument('--permute', default=False,
                        action='store_true',
                        help='run also the "permuted features" experiment')
    parser.add_argument('--seed', default=42,
                        help='random seed to generate hyperparams')

    args = parser.parse_args()
    
    if args.permute:    
        print("permuting...")
        parse_configs(args.split_folder, args.split, args.feature_set, args.nhead, args.nhid, args.nlayers, args.learning_rate, args.dropout, True, args.output_dir, seed=42)
    else:
        parse_configs(args.split_folder, args.split, args.feature_set, args.nhead, args.nhid, args.nlayers, args.learning_rate, args.dropout, False, args.output_dir, seed=42)

        
if __name__ == "__main__":
    main()
