'''Manage hyperparams of the experiment'''
import json
import argparse
from pathlib import Path
import hashlib
import numpy as np


def get_head_opts(feature_set):
    '''Returns a list of valid number of heads for a feature set'''
    if feature_set == 'crispr_orf':
        n_head_opts = [2, 4, 5, 10]
    elif feature_set == 'orf':
        n_head_opts = [2, 4, 8, 11]
    elif feature_set == 'crispr':
        n_head_opts = [1, 2, 4, 8]
    return n_head_opts


def hash_conf(config):
    '''Get a hash value for a given config'''
    utf8_encoded = json.dumps(config, sort_keys=True).encode('utf-8')
    data_md5 = hashlib.md5(utf8_encoded).hexdigest()
    return data_md5


def generate_configs(split, feature_set, permute, num_configs, output_dir, seed=42):
    '''
    Generate random hyperparams configurations
    '''
    rng = np.random.default_rng(seed)
    for _ in range(num_configs):
        learning_rate = float(rng.uniform(10.**-5., 10.**-2.))
        dropout = float(rng.uniform(0.01, 0.4))
        nhid = int(rng.choice([512, 1024, 2048]))
        nhead = int(rng.choice(get_head_opts(feature_set)))
        init_range = float(rng.uniform(10. ** -3, 10. ** -1.))
        config = {'split': split, 'feature_set': feature_set,
                  'batch_size': 1024,
                  'nhead': nhead,
                  'nhid': nhid,
                  'nlayers': 2,
                  'dropout': dropout,
                  'learning_rate': learning_rate,
                  'epochs': 200,
                  'criteria': 'auprc',
                  'permute': permute,
                  'init_range': init_range,
                  'stop_cutoff': 100}
        hash_id = hash_conf(config)
        config['hash_id'] = hash_id
        write_config(config, output_dir)


def write_config(config, output_dir):
    '''Write config'''
    path = Path(output_dir) / config['hash_id']
    path.mkdir(parents=True, exist_ok=True)
    config['model_path'] = str(path / 'model.pt')
    config_path = path / 'config.json'
    if config_path.exists():
        raise ValueError('File already exists')
    with config_path.open('w', encoding='utf8') as fwrite:
        json.dump(config, fwrite)
        


def main():
    '''Parse arguments from command line'''
    parser = argparse.ArgumentParser(
        description='generate config files with different hyperparams')
    parser.add_argument('split', type=str,
                        choices=['compound', 'gene', 'pair'],
                        help='experimental split')
    parser.add_argument('feature_set', type=str,
                        choices=['crispr', 'crispr_orf', 'orf'],
                        help='features used to train')
    parser.add_argument('num_configs', type=int,
                        help='number of random configs to generate')
    parser.add_argument('output_dir',
                        help='path to store the logs and predictions')
    parser.add_argument('--permute', default=False,
                        action='store_true',
                        help='run also the "permuted features" experiment')
    parser.add_argument('--seed', default=42,
                        help='random seed to generate hyperparams')

    args = parser.parse_args()
    
    if args.permute:
        generate_configs(args.split, args.feature_set, True, args.num_configs,
                                args.output_dir, args.seed)
    else:
        generate_configs(args.split, args.feature_set, False, args.num_configs,
                        args.output_dir, args.seed)
                

if __name__ == "__main__":
    main()
