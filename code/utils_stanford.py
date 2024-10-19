"""
Collection of functions for dataset formulation and ML model training/evaluation.

This module presents an end-to-end pipeline for transforming Level 4 CellProfiler
morphological profiles into ML train/test sets, training standard ML models on
these sets, and evaluating ML models.

    Typical usage example:
    
    experiments = utils_stanford.get_standard_experiments()
    X_train, X_test, y_train, y_test = utils_stanford.get_dataset(
        experiments, split='naive', verbose=True)
    gbm, y_proba = utils_stanford.train_standard_model(
        'LightGBM', X_train, X_test, y_train, y_test)
    utils_stanford.eval_trained_model('LightGBM', y_test, y_proba)
    utils_stanford.bootstrap_confidence_interval(y_test, y_proba, conf=0.95, N=1000)
    
For more typical usage examples, see sample_stanford.py.
"""

np_seed = 3
skl_seed = 3

import copy
import os
import random

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

import data_utils


def gene_compound_map():
    """
    Gets mapping between compounds and their gene targets.
    """
    df = pd.read_csv('../data/JUMP-Target_compounds_crispr_orf_connections.csv')
    return df

def get_standard_experiments():
    """
    Gets all U2OS 24-hour compound, 144-hour CRISPR, and 48-hour ORF plates.
    
    Returns:
        A dict of dicts containing the barcode for all
        U2OS 24-hour compound plates, 144-hour CRISPR plates,
        and 48-hour ORF plates, split by modality.
    """
    experiments = {
        'Compound':{
            'BR00116995':'U2OS 24-hour Compound Plate 1',
            'BR00117024':'U2OS 24-hour Compound Plate 2',
            'BR00117025':'U2OS 24-hour Compound Plate 3',
            'BR00117026':'U2OS 24-hour Compound Plate 4',
        },
        'CRISPR':{
            'BR00116997':'U2OS 144-hour CRISPR Plate 1',
            'BR00116998':'U2OS 144-hour CRISPR Plate 2',
            'BR00116999':'U2OS 144-hour CRISPR Plate 3',
            'BR00116996':'U2OS 144-hour CRISPR Plate 4',
        },
        'ORF':{
            'BR00117022':'U2OS 48-hour ORF Plate 1',
            'BR00117023':'U2OS 48-hour ORF Plate 2',
        },
    }
    return experiments

def get_crispr_experiments():
    experiments = {
        'Compound':{
            'BR00116995':'U2OS 24-hour Compound Plate 1',
            'BR00117024':'U2OS 24-hour Compound Plate 2',
            'BR00117025':'U2OS 24-hour Compound Plate 3',
            'BR00117026':'U2OS 24-hour Compound Plate 4',
        },
        'CRISPR':{
            'BR00116997':'U2OS 144-hour CRISPR Plate 1',
            'BR00116998':'U2OS 144-hour CRISPR Plate 2',
            'BR00116999':'U2OS 144-hour CRISPR Plate 3',
            'BR00116996':'U2OS 144-hour CRISPR Plate 4',
        },
    }
    return experiments

def get_orf_experiments():
    experiments = {
        'Compound':{
            'BR00116995':'U2OS 24-hour Compound Plate 1',
            'BR00117024':'U2OS 24-hour Compound Plate 2',
            'BR00117025':'U2OS 24-hour Compound Plate 3',
            'BR00117026':'U2OS 24-hour Compound Plate 4',
        },
        'ORF':{
            'BR00117022':'U2OS 48-hour ORF Plate 1',
            'BR00117023':'U2OS 48-hour ORF Plate 2',
        },
    }
    return experiments
    
def get_raw_dataframe(experiments, filetype='normalized_feature_select_negcon.csv.gz'):
    """
    Concatenates the morphological profiles of the specified experiments into one DataFrame.
    
    Args:
        experiments: A dict of dicts specifying which experiments to consider.
            See utils_stanford.get_standard_experiments.
    
    Returns:
        A DataFrame where each row corresponds to the morphological profile of
        a compound, CRISPR, or ORF experiment. The inner join is taken, so each
        column in the DataFrame is a feature present in all morphological
        profiles. Includes metadata columns. By default, the features are
        Level 4, i.e. already normalized and feature selected.
    """
    df = pd.DataFrame()
    experiment_name = 'CPJUMP1'
    
    for modality in experiments:
        for plate in experiments[modality]:
            data_df = (
                data_utils.load_data(experiment_name, plate,
                                filetype)
                .assign(Metadata_modality=modality)
                .rename(columns={'Metadata_target':'Metadata_genes'})
            )
            data_df = data_utils.remove_negcon_empty_wells(data_df)

            if df.shape[0] == 0:
                df = data_df.copy()
            else:
                frames = [df, data_df]
                df = pd.concat(frames, ignore_index=True, join='inner')
    
    return df

def get_collated_profiles(experiments, filetype='normalized_feature_select_outlier_trimmed_negcon.parquet'):
    df = pd.DataFrame()
    path = f'../pilot-cpjump1-data/collated/2020_11_04_CPJUMP1/2020_11_04_CPJUMP1_all_{filetype}'
    all_df = pd.read_parquet(path, engine='fastparquet')  # dataframe with features for all plates
    experiment_name = 'CPJUMP1'
    
    for modality in experiments:
        for plate in experiments[modality]:
            data_df = (
                all_df.loc[all_df['Metadata_Plate'] == plate]
                .assign(Metadata_modality=modality)
            )
            data_df = data_utils.remove_negcon_empty_wells(data_df)

            if df.shape[0] == 0:
                df = data_df.copy()
            else:
                frames = [df, data_df]
                df = pd.concat(frames, ignore_index=True, join='inner')
    
    return df

def get_median_consensus_profiles(df, modality):
    """
    Extracts median consensus profiles for a given modality from a larger DataFrame.
    
    Args:
        df: A DataFrame containing morphological profiles with metadata for
            some compound, CRISPR, and/or ORF experiments.
            See utils_stanford.get_raw_dataframe.
        modality: Which modality to extract median consensus features for.
            Must be one of 'Compound', 'CRISPR', or 'ORF'.
    
    Returns:
        A DataFrame where the median is taken for each feature over all
        replicate plates within a given modality. For example, if the compound
        modality is specified, each row corresponds to a distinct compound.
    """
    if modality == 'Compound':
        gene_compound_df = gene_compound_map()[['pert_iname_compound', 'pert_id_compound']]
        gene_compound_df.columns = ['Metadata_pert_iname_compound', 'Metadata_pert_id_compound']
        modality_df = df.query('Metadata_modality=="Compound"').copy()
        # Truncate the last 9 characters in each string to match the format of gene_compound_df
        modality_df['Metadata_broad_sample'] = modality_df['Metadata_broad_sample'].str.slice(0,-9)
        modality_df = modality_df.merge(gene_compound_df, how='inner',
                                        left_on='Metadata_broad_sample',
                                        right_on='Metadata_pert_id_compound')
    elif modality == 'CRISPR':
        modality_df = df.query('Metadata_modality=="CRISPR"')
    elif modality == 'ORF':
        modality_df = df.query('Metadata_modality=="ORF"')
    
    metadata_df = (
        data_utils.get_metadata(modality_df)
        .drop_duplicates(subset=['Metadata_broad_sample'])
    )
    modality_df = modality_df.groupby(['Metadata_broad_sample']).median().reset_index()
    modality_df = (
        metadata_df.merge(modality_df, on='Metadata_broad_sample')
        .drop(columns=['Metadata_Well'])
    )
    
    return modality_df

def get_cross_dataframe(modality_dfs):
    """
    Generates cross dataframe with concatenated compound and gene features.
    
    Args:
        modality_dfs: A dict specifying which modalities the data includes.
            Key is one of 'Compound', 'CRISPR', or 'ORF'.
            Value is a DataFrame containing modality-specific features.
    
    Returns:
        A DataFrame with the concatenated compound, CRISPR, and ORF
        features for every possible compound-gene pair. Includes 'pair'
        column specifying whether the pair is a true pair.
    """
    if 'CRISPR' in modality_dfs and 'ORF' in modality_dfs:
        gene_df = modality_dfs['CRISPR'].join(modality_dfs['ORF'].set_index(
            'Metadata_genes_ORF'), on='Metadata_genes_CRISPR')
    elif 'CRISPR' in modality_dfs:
        gene_df = modality_dfs['CRISPR']
    elif 'ORF' in modality_dfs:
        gene_df = modality_dfs['ORF']
    
    cross_df = modality_dfs['Compound'].merge(gene_df, how='cross')
    cross_df['pair'] = 0
    
    gene_compound_df = gene_compound_map()
    
    if 'CRISPR' in modality_dfs:  # CRISPR or CRISPR+ORF features
        for i in range(len(gene_compound_df)):
            compound_id = gene_compound_df.loc[i,'pert_id_compound']
            crispr_id = gene_compound_df.loc[i,'broad_sample_crispr']
            cross_df.loc[
                (cross_df.Metadata_broad_sample_Compound == compound_id) &
                (cross_df.Metadata_broad_sample_CRISPR == crispr_id),
                'pair'
            ] = 1
    
    else:  # Just ORF features
        for i in range(len(gene_compound_df)):
            compound_id = gene_compound_df.loc[i,'pert_id_compound']
            orf_id = gene_compound_df.loc[i,'broad_sample_orf']
            cross_df.loc[
                (cross_df.Metadata_broad_sample_Compound == compound_id) &
                (cross_df.Metadata_broad_sample_ORF == orf_id),
                'pair'
            ] = 1
    
    return cross_df
    
def get_naive_dataset(cross_df, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Constructs train/val/test sets with naive random split.
    
    Args:
        cross_df: A DataFrame with the concatenated compound, CRISPR, and ORF
            features for every possible compound-gene pair. Includes 'pair'
            column specifying whether the pair is a true pair.
        train_size: Optional; The proportion of the dataset to use for training.
        val_size: Optional; The proportion of the dataset to use for validation.
        test_size: Optional; The proportion of the dataset to user for testing.
    
    Returns:
        Six DataFrames: X_train, X_val, X_test, y_train, y_val, y_test. These arrays
        correspond to the training features, validation features, test features,
        training labels (0 or 1 for whether the compound and CRISPR/ORF have the same gene
        target), validation labels, and test labels, respectively. X arrays include metadata.
    """
    if train_size + val_size + test_size != 1.:
        raise ValueError(
            f'train_size={train_size}, val_size={val_size}, test_size={test_size} do not add up to 1.')
        
    X = cross_df.drop('pair', axis=1)
    y = cross_df['pair']
    X_train, X_non_train, y_train, y_non_train = sklearn.model_selection.train_test_split(
        X, y, test_size=val_size+test_size, random_state=42)
    
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
        X_non_train, y_non_train, test_size=test_size/(val_size+test_size), random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_leave_out_compound_dataset(cross_df, modality_dfs, train_size=0.6, val_size=0.2, test_size=0.2):
    if train_size + val_size + test_size != 1.:
        raise ValueError(
            f'train_size={train_size}, val_size={val_size}, test_size={test_size} do not add up to 1.')
        
    np.random.seed(np_seed)
    all_compounds = list(
        modality_dfs['Compound'].Metadata_pert_iname_compound_Compound.drop_duplicates())
    
    train_compounds = np.random.choice(all_compounds,
                                   round(len(all_compounds) * train_size),
                                   False)
    
    non_train_compounds = [compound for compound in all_compounds if compound not in train_compounds]
    
    val_compounds = np.random.choice(non_train_compounds,
                                   round(len(non_train_compounds) * val_size/(val_size + test_size)),
                                   False)
    
    test_compounds = [compound for compound in non_train_compounds if compound not in val_compounds]
    
    train_df = cross_df[cross_df['Metadata_pert_iname_compound_Compound']
                        .isin(train_compounds)]
    val_df = cross_df[cross_df['Metadata_pert_iname_compound_Compound']
                        .isin(val_compounds)]
    test_df = cross_df[cross_df['Metadata_pert_iname_compound_Compound']
                        .isin(test_compounds)]

    X_train = train_df.drop('pair', axis=1)
    y_train = train_df['pair']
    X_val = val_df.drop('pair', axis=1)
    y_val = val_df['pair']
    X_test = test_df.drop('pair', axis=1)
    y_test = test_df['pair']
    
    return X_train, X_val, X_test, y_train, y_val, y_test
         
def get_leave_out_one_dataset(cross_df, modality_dfs, split, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Constructs train/test sets with distinct CRISPRs or ORFs.
    
    Args:
        cross_df: A DataFrame with the concatenated compound, CRISPR, and ORF
            features for every possible compound-gene pair. Includes 'pair'
            column specifying whether the pair is a true pair.
        modality_dfs: A dict specifying which modalities the data includes.
            Key is one of 'Compound', 'CRISPR', or 'ORF'.
            Value is a DataFrame containing modality-specific features.
        split: How to split the train and test sets. 'CRISPR' ensures no
            overlap between CRISPRs seen in training and in testing,
            where proportion train_size of compounds are held out for testing.
            'ORF' is analogous.
        train_size: Optional; The proportion of the dataset to use for training.
    
    Returns:
        Six DataFrames: X_train, X_val, X_test, y_train, y_val, y_test. These arrays
        correspond to the training features, validation features, test features,
        training labels (0 or 1 for whether the compound and CRISPR/ORF have the same gene
        target), validation labels, and test labels, respectively. X arrays include metadata.
    """
    if train_size + val_size + test_size != 1.:
        raise ValueError(
            f'train_size={train_size}, val_size={val_size}, test_size={test_size} do not add up to 1.')
        
    np.random.seed(np_seed)
    if split == 'CRISPR':
        all_perts = list(
            modality_dfs['CRISPR'].Metadata_broad_sample_CRISPR.drop_duplicates())
    elif split == 'ORF':
        all_perts = list(
            modality_dfs['ORF'].Metadata_broad_sample_ORF.drop_duplicates())
    else:
        raise ValueError(
            f'{split} split is not currently supported. Must be "Compound", "CRISPR", or "ORF".')
    
    train_perts = np.random.choice(all_perts,
                                   round(len(all_perts) * train_size),
                                   False)
    non_train_perts = [pert for pert in all_perts if pert not in train_perts]
    val_perts = np.random.choice(non_train_perts,
                                   round(len(non_train_perts) * val_size/(val_size + test_size)),
                                   False)
    
    test_perts = [pert for pert in non_train_perts if pert not in val_perts]
    train_df = cross_df[cross_df['Metadata_broad_sample_'+split]
                       .isin(train_perts)]
    val_df = cross_df[cross_df['Metadata_broad_sample_'+split]
                       .isin(val_perts)]
    test_df = cross_df[cross_df['Metadata_broad_sample_'+split]
                       .isin(test_perts)]
    
    X_train = train_df.drop('pair', axis=1)
    y_train = train_df['pair']
    X_val = val_df.drop('pair', axis=1)
    y_val = val_df['pair']
    X_test = test_df.drop('pair', axis=1)
    y_test = test_df['pair']
    
    return X_train, X_val, X_test, y_train, y_val, y_test
        
def get_leave_out_together_dataset(cross_df, modality_dfs, split, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Constructs train/test sets where two compounds/sgRNAs with the same gene target are
    always placed in the same set.
    
    Args:
        cross_df: A DataFrame with the concatenated compound, CRISPR, and ORF
            features for every possible compound-gene pair. Includes 'pair'
            column specifying whether the pair is a true pair.
        modality_dfs: A dict specifying which modalities the data includes.
            Key is one of 'Compound', 'CRISPR', or 'ORF'.
            Value is a DataFrame containing modality-specific features.
        split: How to split the train and test sets. 'CRISPR_together' or 'Compound_together'
        train_size: Optional; The proportion of the dataset to use for training.
    
    Returns:
        Six DataFrames: X_train, X_val, X_test, y_train, y_val, y_test. These arrays
        correspond to the training features, validation features, test features,
        training labels (0 or 1 for whether the compound and CRISPR/ORF have the same gene
        target), validation labels, and test labels, respectively. X arrays include metadata.
    """
    if train_size + val_size + test_size != 1.:
        raise ValueError(
            f'train_size={train_size}, val_size={val_size}, test_size={test_size} do not add up to 1.')
        
    np.random.seed(np_seed)
    if split == 'CRISPR_together':
        split = 'CRISPR'
    elif split == 'Compound_together':
        split = 'Compound'
        
    all_genes = list(modality_dfs['Compound'].Metadata_genes_Compound.drop_duplicates())
    
    train_genes = np.random.choice(all_genes,
                                   round(len(all_genes) * train_size),
                                   False)
    
    non_train_genes = [gene for gene in all_genes if gene not in train_genes]
    val_genes = np.random.choice(non_train_genes,
                                   round(len(non_train_genes) * val_size/(val_size + test_size)),
                                   False)
    
    test_genes = [gene for gene in non_train_genes if gene not in val_genes]
    
    train_df = cross_df[cross_df['Metadata_genes_'+split]
                       .isin(train_genes)]
    val_df = cross_df[cross_df['Metadata_genes_'+split]
                       .isin(val_genes)]
    test_df = cross_df[cross_df['Metadata_genes_'+split]
                       .isin(test_genes)]
    
    X_train = train_df.drop('pair', axis=1)
    y_train = train_df['pair']
    X_val = val_df.drop('pair', axis=1)
    y_val = val_df['pair']
    X_test = test_df.drop('pair', axis=1)
    y_test = test_df['pair']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_leave_out_pair_dataset(cross_df, modality_dfs, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Constructs train/test sets with non-overlapping gene-compound pairs
    
    Args:
        cross_df: A DataFrame with the concatenated compound, CRISPR, and ORF
            features for every possible compound-gene pair. Includes 'pair'
            column specifying whether the pair is a true pair.
        modality_dfs: A dict specifying which modalities the data includes.
            Key is one of 'Compound', 'CRISPR', or 'ORF'.
            Value is a DataFrame containing modality-specific features.
        train_size: Optional; The proportion of the dataset to use for training.
    
    Returns:
        Six DataFrames: X_train, X_val, X_test, y_train, y_val, y_test. These arrays
        correspond to the training features, validation features, test features,
        training labels (0 or 1 for whether the compound and CRISPR/ORF have the same gene
        target), validation labels, and test labels, respectively. X arrays include metadata.
    """
    if train_size + val_size + test_size != 1.:
        raise ValueError(
            f'train_size={train_size}, val_size={val_size}, test_size={test_size} do not add up to 1.')
        
    np.random.seed(np_seed)
    if 'CRISPR' in modality_dfs:
        all_pairs = cross_df[['Metadata_genes_CRISPR',
                          'Metadata_pert_iname_compound_Compound']].drop_duplicates().reset_index(drop=True)
    else:
        all_pairs = cross_df[['Metadata_genes_ORF',
                          'Metadata_pert_iname_compound_Compound']].drop_duplicates().reset_index(drop=True)
    
    train_pairs, non_train_pairs = sklearn.model_selection.train_test_split(
        all_pairs, test_size=val_size+test_size, random_state=skl_seed)
    
    val_pairs, test_pairs = sklearn.model_selection.train_test_split(
        non_train_pairs, test_size=test_size/(val_size+test_size), random_state=skl_seed)
    
    train_df = cross_df.merge(train_pairs)
    val_df = cross_df.merge(val_pairs)
    test_df = cross_df.merge(test_pairs)
    
    X_train = train_df.drop('pair', axis=1)
    y_train = train_df['pair']
    X_val = val_df.drop('pair', axis=1)
    y_val = val_df['pair']
    X_test = test_df.drop('pair', axis=1)
    y_test = test_df['pair']
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def make_data_split(experiments, folder, collated=False, split='naive',
                filetype='normalized_feature_select_negcon.csv.gz',
                train_size=0.6, val_size=0.2, test_size=0.2, verbose=False):
    """
    Constructs train/test sets suited for binary classification machine learning.
    
    Args:
        experiments: A dict of dicts specifying which experiments to consider.
            See utils_stanford.get_standard_experiments.
        split: Optional; How to split the train and test sets. 'naive' is a
            random split. 'Compound' ensures no overlap between compounds
            seen in training and in testing, where proportion train_size of
            compounds are held out for testing. Analogously, 'CRISPR' ensures
            no overlap between CRISPRs seen in training and in testing. 'ORF' is
            analogous. 'Compound+CRISPR' ensures that the train and test sets
            have both distinct compounds AND distinct CRISPRs. 'Compound+ORF'
            ensures distinct compounds and ORFs. Note that 'Compound+CRISPR' and
            'Compound+ORF' options necessarily discard the train compound + test
            gene and test compound + train gene samples.
        train_size: Optional; The proportion of the dataset to use for training.
            For 'Compound+CRISPR' or 'Compound+ORF' splits, this refers to the
            proportion of the pruned dataset to use for training.
        verbose: Optional; If True, prints out additional information on the
            shape of the train and test sets, as well as the number of
            positive pairs in both sets.
    
    Returns:
        Four DataFrames: X_train, X_test, y_train, y_test. These arrays
        correspond to the training features, test features, training labels
        (0 or 1 for whether the compound and CRISPR/ORF have the same gene
        target), and test labels, respectively. In all cases, the test set
        contains approximately train_size of the total data across both train
        and test sets.
    """
    if collated:
        df = get_collated_profiles(experiments)
    else:
        df = get_raw_dataframe(experiments, filetype)
    
    # Split the DataFrame into Compound and CRISPR/ORF DataFrames
    modality_dfs = {}  # modality_dfs['Compound'] contains the Compound df.
    for modality in experiments:
        modality_df = get_median_consensus_profiles(df, modality)
        modality_df.columns = ['{}_{}'.format(col, modality) for col in modality_df.columns]
        modality_dfs[modality] = modality_df
    
    cross_df = get_cross_dataframe(modality_dfs)
     
    if split == 'naive':  # naive train/test split
        X_train, X_val, X_test, y_train, y_val, y_test = get_naive_dataset(cross_df, train_size, val_size, test_size)
    
    elif split == 'Compound':  # leave out compound (using pert_iname)
        X_train, X_val, X_test, y_train, y_val, y_test = get_leave_out_compound_dataset(cross_df, modality_dfs, train_size, val_size, test_size)
        
    # leave out one of Compound, CRISPR, or ORF
    elif split in ['CRISPR', 'ORF']:
        X_train, X_val, X_test, y_train, y_val, y_test = get_leave_out_one_dataset(cross_df, modality_dfs, split, train_size, val_size, test_size)
    
#     elif split == 'target':
#         X_train, X_val, X_test, y_train, y_val, y_test = get_leave_out_target_dataset(cross_df, modality_dfs, split, train_size, val_size, test_size)
    
    elif split in ['Compound_together','CRISPR_together']:  # Compounds/CRISPRs with the same target are kept together
        X_train, X_val, X_test, y_train, y_val, y_test = get_leave_out_together_dataset(cross_df, modality_dfs, split, train_size, val_size, test_size)
    
    elif split == 'pair':
        X_train, X_val, X_test, y_train, y_val, y_test = get_leave_out_pair_dataset(cross_df, modality_dfs, train_size, val_size, test_size)
    
    else:
        raise ValueError(
            f'{split} split is not currently supported. See docstring for supported train/test splits.')
        
    if verbose:
        print('X_train shape: {}'.format(X_train.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('Num. positive pairs in train set: {}'.format(sum(y_train)))
        print('X_val shape: {}'.format(X_val.shape))
        print('y_val shape: {}'.format(y_val.shape))
        print('Num. positive pairs in val set: {}'.format(sum(y_val)))
        print('X_test shape: {}'.format(X_test.shape))
        print('y_test shape: {}'.format(y_test.shape))
        print('Num. positive pairs in test set: {}'.format(sum(y_test)))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data_split(cross_df, modality_dfs, folder, split_type):
    train_pairs = pd.read_csv(f'{folder}/{split_type}__train.csv')
    val_pairs = pd.read_csv(f'{folder}/{split_type}__val.csv')
    test_pairs = pd.read_csv(f'{folder}/{split_type}__test.csv')
    
    if 'CRISPR' in modality_dfs:
        cols = ['Metadata_genes_CRISPR', 'Metadata_pert_iname_compound_Compound']
    else:
        cols = ['Metadata_genes_ORF', 'Metadata_pert_iname_compound_Compound']
    
    train_pairs.columns = cols
    val_pairs.columns = cols
    test_pairs.columns = cols
    
    train_df = cross_df.merge(train_pairs)
    val_df = cross_df.merge(val_pairs)
    test_df = cross_df.merge(test_pairs)
    
    X_train = train_df.drop('pair', axis=1)
    y_train = train_df['pair']
    X_val = val_df.drop('pair', axis=1)
    y_val = val_df['pair']
    X_test = test_df.drop('pair', axis=1)
    y_test = test_df['pair']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_split_from_folder(experiments, folder, collated=False, split_type='pair',
                filetype='normalized_feature_select_negcon.csv.gz', permute=False):
    if collated:
        df = get_collated_profiles(experiments)
    else:
        df = get_raw_dataframe(experiments, filetype)
    
    # Split the DataFrame into Compound and CRISPR/ORF DataFrames
    modality_dfs = {}  # modality_dfs['Compound'] contains the Compound df.
    for modality in experiments:
        modality_df = get_median_consensus_profiles(df, modality)
        modality_df.columns = ['{}_{}'.format(col, modality) for col in modality_df.columns]
        modality_dfs[modality] = modality_df
    
    if permute and 'ORF' in modality_dfs:
        from permute import permute_orf_feats
        modality_dfs['ORF'] = permute_orf_feats(modality_dfs['ORF'])

    if permute and 'CRISPR' in modality_dfs:
        from permute import permute_crispr_feats
        modality_dfs['CRISPR'] = permute_crispr_feats(modality_dfs['CRISPR'])

    cross_df = get_cross_dataframe(modality_dfs)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_split(
        cross_df, modality_dfs, folder, split_type)
    return X_train, X_val, X_test, y_train, y_val, y_test
     
def precision_at_top_K(y_test, y_proba, K, verbose=False):
    """
    Calculates the precision at top K.
    
    Args:
        y_test: A NumPy array containing the true test set labels (0 or 1).
        y_proba: A NumPy array containing the model's predicted label
            probabilities (0 < p < 1). For SVM models this is the decision
            function evaluated on each sample so may not be a probability.
            See utils_stanford.train_standard_model.
        K: The value of K to calculate precision at top K for.
        verbose: Optional; If True, prints whether each of the top K predicted
            pairs is a true pair.
    
    Returns:
        The precision at top K for the given true labels and predicted
        probabilities, that is the proportion of the top K predicted pairs
        which represent true pairs.
    """
    # indices of the top K most likely to be pairs
    top_K_indices = y_proba.argsort()[-K:][::-1]
    if verbose:
        print(y_test[top_K_indices])
    # how many of those top K are actually pairs    
    p_at_K = sum(y_test[top_K_indices])/K
    return p_at_K

def enrichment_at_top_K(y_test, y_proba, K, verbose=False):
    """
    Calculates enrichment at top K.
    
    Args:
        y_test: A NumPy array containing the true test set labels (0 or 1).
        y_proba: A NumPy array containing the model's predicted label
            probabilities (0 < p < 1). For SVM models this is the decision
            function evaluated on each sample so may not be a probability.
            See utils_stanford.train_standard_model.
        K: The value of K to calculate precision at top K for.
        verbose: Optional; If True, prints whether each of the top K predicted
            pairs is a true pair.
    
    Returns:
        The enrichment at top K for the given true labels and predicted
        probabilities, that is the proportion of the top K predicted pairs
        which represent true pairs divided by the proportion of the bottom N-K
        predicted pairs which represent true pairs.
    """
    # indices of the top K most likely to be pairs
    top_K_indices = y_proba.argsort()[-K:][::-1]
    bottom_indices = y_proba.argsort()[:-K][::-1]
    if verbose:
        print(y_test[top_K_indices])
    # how many of those top K are actually pairs    
    p_at_K = sum(y_test[top_K_indices])/K
    # how many of those bottom N-K are actually pairs
    N = len(y_test)
    p_bottom = sum(y_test[bottom_indices])/(N - K)
    return p_at_K/p_bottom

def plot_roc_curve(fpr, tpr, label=None):
    """
    Plots the ROC curve for a machine learning model.
    
    Args:
        fpr: NumPy array of increasing false positive rates such that element i
            is the false positive rate of predictions with
            score >= thresholds[i]. See sklearn.metrics.roc_curve.
        tpr: NumPy array of increasing true positive rates such that element i
            is the false positive rate of predictions with
            score >= thresholds[i]. See sklearn.metrics.roc_curve.
        label: Label to use on the ROC plot, typically the name of the model.
    """
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], color='navy',linestyle = '--') # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - ' + label)
    plt.legend(loc='lower right')
    plt.show()
  
def roc_metrics(label, y_test, y_proba):
    """
    Plots ROC curve and computes area under the ROC curve (AUROC).
    
    Args:
        label: Label to use on the ROC plot, typically the name of the model.
        y_test: A NumPy array containing the true test set labels (0 or 1).
        y_proba: A NumPy array containing the model's predicted label
            probabilities (0 < p < 1). For SVM models this is the decision
            function evaluated on each sample so may not be a probability.
            See utils_stanford.train_standard_model.
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_proba)
    plot_roc_curve(fpr, tpr, label)
    print ('AUROC: ', sklearn.metrics.roc_auc_score(y_test, y_proba))
    
def eval_trained_model(label, y_test, y_proba):
    """
    Computes the AUROC, AUPRC, and precision at top K for a model.
    
    Args:
        label: Label to use on the ROC plot, typically the name of the model.
        y_test: A NumPy array containing the true test set labels (0 or 1).
        y_proba: A NumPy array containing the model's predicted label
            probabilities (0 < p < 1). For SVM models this is the decision
            function evaluated on each sample so may not be a probability.
            See utils_stanford.train_standard_model.
    """
    print(label)
    print('-----------------------------')
    roc_metrics(label, y_test, y_proba)
    print ('AUPRC: ', sklearn.metrics.average_precision_score(y_test, y_proba))
    
    K = sum(y_test)  # number of positive pairs in y_test
    # We refer to this metric as precision at top prevalence, since K equals
    # the number of positive pairs in the test set.
    print('P@Top{}: {}'.format(K, precision_at_top_K(y_test, y_proba, K)))
    
    K = 10
    print('P@Top{}: {}'.format(K, precision_at_top_K(y_test, y_proba, K)))
    
    K = 50
    print('P@Top{}: {}'.format(K, precision_at_top_K(y_test, y_proba, K)))
    
    K = 100
    print('P@Top{}: {}'.format(K, precision_at_top_K(y_test, y_proba, K)))
    
def bootstrap_confidence_interval(y_test, y_proba, conf=0.95, N=1000):
    """
    Constructs bootstrapped confidence intervals for AUROC, AUPRC, and P@TopPrev metrics.
    
    Args:
        y_test: A NumPy array containing the true test set labels (0 or 1).
        y_proba: A NumPy array containing the model's predicted label
            probabilities (0 < p < 1). For SVM models this is the decision
            function evaluated on each sample so may not be a probability.
            See utils_stanford.train_standard_model.
        conf: Optional; The confidence level, typically 95%.
        N: Optional; The number of resamples to perform during the bootstrap.
            Use N=10000 if speed is not an issue.
    """
    aurocs = []
    auprcs = []
    p_at_top_prev = [] # precision at top K (K = number of positive pairs in test set)
    enrichment_at_top_prev = []  # enrichment at top K
    num_positive_pairs = sum(y_test)
    
    np.random.seed(42)
    
    for i in range(N):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        by_test = copy.deepcopy(y_test[idx])
        by_proba = copy.deepcopy(y_proba[idx])
        aurocs.append(sklearn.metrics.roc_auc_score(by_test, by_proba))
        auprcs.append(sklearn.metrics.average_precision_score(by_test, by_proba))
        p_at_top_prev.append(precision_at_top_K(by_test, by_proba, num_positive_pairs))
        enrichment_at_top_prev.append(enrichment_at_top_K(by_test, by_proba, num_positive_pairs))
    
    aurocs = np.sort(aurocs)
    auprcs = np.sort(auprcs)
    p_at_top_prev = np.sort(p_at_top_prev)
    enrichment_at_top_prev = np.sort(enrichment_at_top_prev)
    
    # 95% confidence interval by default
    lo = (1.0 - conf)/2
    hi = conf + lo
    
    print('AUROC Bootstrap {}% Confidence Interval: {} ({}, {})'.format(
        round(conf * 100), round(np.mean(aurocs), 3),
        round(aurocs[round(lo * N)], 3), round(aurocs[round(hi * N)], 3),
    ))
    print('AUPRC Bootstrap {}% Confidence Interval: {} ({}, {})'.format(
        round(conf * 100), round(np.mean(auprcs), 3),
        round(auprcs[round(lo * N)], 3), round(auprcs[round(hi * N)], 3),
    ))
    print('P@TopPrev Bootstrap {}% Confidence Interval: {} ({}, {})'.format(
        round(conf * 100), round(np.mean(p_at_top_prev), 3),
        round(p_at_top_prev[round(lo * N)], 3), round(p_at_top_prev[round(hi * N)], 3),
    ))
    print('Enrichment@TopPrev Bootstrap {}% Confidence Interval: {} ({}, {})'.format(
        round(conf * 100), round(np.mean(enrichment_at_top_prev), 3),
        round(enrichment_at_top_prev[round(lo * N)], 3), round(enrichment_at_top_prev[round(hi * N)], 3),
    ))
    
