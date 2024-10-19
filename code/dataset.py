'''Functions to train and evaluate transformers in gene compound matching task'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils_stanford
import data_utils

class CellPaintingDataset(Dataset):
    """Cell Painting Image Profile dataset."""

    def __init__(self, X, y, nmodal):
        # features
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        # labels (0 or 1)
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        self.nmodal = nmodal
        self.nfeatures = X.shape[1]//self.nmodal

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feats = self.X[idx]
        feats = self.group_features(feats)
        label = self.y[idx]
        return feats, label

    def group_features(self, features):
        """
        (300,) feature vector is transformed into (3, 100) tensor.
        out[0] = [in[0], ..., in[99]]
        Each row corresponds to 100 features, and the 3 rows correspond to
        compound, CRISPR, and ORF, respectively.
        """
        return torch.reshape(features, (self.nmodal, self.nfeatures))

    def get_loader(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class CellPaintingDatasetWithMeta(CellPaintingDataset):

    def __init__(self, X, y, nmodal, meta):
        super().__init__(X, y, nmodal)
        self.meta = meta.values.tolist()
        self.meta_cols = meta.columns

    def __getitem__(self, idx):
        feats, label = super().__getitem__(idx)
        meta = self.meta[idx]
        return feats, label, meta

def get_dataset_params(feature_set):
    '''Get dataset configuration'''
    if feature_set == 'crispr_orf':
        experiments = utils_stanford.get_standard_experiments()
    elif feature_set == 'crispr':
        experiments = utils_stanford.get_crispr_experiments()
    elif feature_set == 'orf':
        experiments = utils_stanford.get_orf_experiments()
    else:
        raise ValueError(f'Invalid feature set: {feature_set}')

    return experiments

def get_loaders(feature_set, folder, split, batch_size, permute):
    '''Get pytorch data loaders for the train val test subsets'''
    experiments = get_dataset_params(feature_set)
    nmodal = len(experiments)

    data = utils_stanford.load_split_from_folder(
        experiments, folder=folder, split_type=split, permute=permute)
    X_train, X_val, X_test, y_train, y_val, y_test = data

    # Training data
    X_train = data_utils.get_featuredata(X_train)
    training_data = CellPaintingDataset(X_train, y_train, nmodal)
    train_loader = training_data.get_loader(batch_size, shuffle=True)

    # Validation data
    meta = data_utils.get_metadata(X_val).fillna('None')
    X_val = data_utils.get_featuredata(X_val)
    val_data = CellPaintingDatasetWithMeta(X_val, y_val, nmodal, meta)
    val_loader = val_data.get_loader(batch_size, shuffle=False)

    # Test data with metadata to export prediction labels
    meta = data_utils.get_metadata(X_test).fillna('None')
    X_test = data_utils.get_featuredata(X_test)
    test_data = CellPaintingDatasetWithMeta(X_test, y_test, nmodal, meta)
    # disable shuffling so bootstrap is reproducible
    test_loader = test_data.get_loader(batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
