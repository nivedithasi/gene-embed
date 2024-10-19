'''
Functions to permute embedding representation of genes
'''
import numpy as np
import pandas as pd
import data_utils


def gene_compound_map():
    """
    Gets mapping between compounds and their gene targets.
    """
    dframe = pd.read_csv(
        '../data/JUMP-Target_compounds_crispr_orf_connections.csv')
    return dframe


def get_orf_mapper(from_file=True, orf=None, mapper=None):
    '''Get permuted version of ORF ids based on gene mapper'''

    if from_file:
        orf_mapper = pd.read_csv('../permutations/orf_permutation.csv')
        orf_mapper = orf_mapper.set_index('real').permuted
        return orf_mapper

    target_col = 'Metadata_genes_ORF'
    orf_id_col = 'Metadata_broad_sample_ORF'

    target_to_orf = orf.set_index(target_col)[orf_id_col]

    new_target = mapper[orf[target_col]].values
    new_orf = target_to_orf[new_target]
    old_orf = orf[orf_id_col]

    orf_mapper = pd.Series(index=old_orf.values, data=new_orf.values)
    return orf_mapper


def get_crispr_mapper(from_file=True, crispr=None, mapper=None):
    '''Get permuted version of CRISPR ids based on gene mapper'''
    if from_file:
        crispr_mapper = pd.read_csv('../permutations/crispr_permutation.csv')
        crispr_mapper = crispr_mapper.set_index('real').permuted
        return crispr_mapper

    gene_col = 'Metadata_genes_CRISPR'
    crispr_id_col = 'Metadata_broad_sample_CRISPR'
    assert crispr[crispr_id_col].is_unique
    target_to_crispr = crispr.groupby(gene_col)[crispr_id_col].apply(set)
    new_targets = mapper[crispr[gene_col]].values

    new_crispr = []
    for new_target in new_targets:
        item = target_to_crispr[new_target].pop()
        if not target_to_crispr[new_target]:
            # gene has single crispr, keep last one
            target_to_crispr[new_target].add(item)
        new_crispr.append(item)

    old_crispr = crispr[crispr_id_col].values

    crispr_mapper = pd.Series(index=old_crispr, data=new_crispr)
    return crispr_mapper


def permute_genes(from_file=True, seed=42):
    '''
    Get a permuted series of genes. Genes that have same number of
    crispr codes are permuted.
    '''
    if from_file:
        mapper = pd.read_csv('../permutations/gene_permutation.csv')
        mapper = mapper.set_index('real').permuted
        return mapper

    rng = np.random.default_rng(seed=seed)
    matches = gene_compound_map()

    groupby = matches.groupby('gene')
    crispr_count = groupby['broad_sample_crispr'].nunique()

    gene_mapper = {}
    for counts in crispr_count.unique():
        keys = crispr_count[crispr_count == counts].index
        values = rng.permutation(list(keys))
        gene_mapper.update(dict(zip(keys, values)))
    return pd.Series(gene_mapper)


def permute_orf_feats(orf):
    '''Permute ORF feats using gene mapper'''
    feat_cols = data_utils.get_featurecols(orf)
    feats = orf.set_index('Metadata_broad_sample_ORF')[feat_cols]

    # build map from id to new features
    orf_mapper = get_orf_mapper()
    feat_values = feats.loc[orf_mapper.values].values
    feats_mapper = pd.DataFrame(data=feat_values,
                                index=orf_mapper.index, columns=feat_cols)

    # Assign new features
    new_feats = feats_mapper.loc[orf['Metadata_broad_sample_ORF']]
    new_feats = new_feats[feat_cols].values
    orf[feat_cols] = new_feats
    return orf


def permute_crispr_feats(crispr):
    '''Permute CRISPR feats using gene mapper'''
    feat_cols = data_utils.get_featurecols(crispr)
    feats = crispr.set_index('Metadata_broad_sample_CRISPR')[feat_cols]

    # build map from id to new features
    crispr_mapper = get_crispr_mapper()
    feat_values = feats.loc[crispr_mapper.values].values
    feats_mapper = pd.DataFrame(data=feat_values,
                                index=crispr_mapper.index, columns=feat_cols)

    # Assign new features
    new_feats = feats_mapper.loc[crispr['Metadata_broad_sample_CRISPR']]
    new_feats = new_feats[feat_cols].values
    crispr[feat_cols] = new_feats
    return crispr


def generate_permutations(save=False):
    '''Generate permutations to across genes orf and compound ids'''
    modality_dfs = get_modalities()
    mapper = permute_genes(from_file=False)

    crispr_mapper = get_crispr_mapper(False, modality_dfs['CRISPR'], mapper)
    crispr_mapper = crispr_mapper.reset_index()
    crispr_mapper.columns = 'real', 'permuted'

    orf_mapper = get_orf_mapper(False, modality_dfs['ORF'], mapper)
    orf_mapper = orf_mapper.reset_index()
    orf_mapper.columns = 'real', 'permuted'

    mapper = mapper.reset_index()
    mapper.columns = 'real', 'permuted'

    if save:
        crispr_mapper.to_csv('../permutations/crispr_permutation.csv')
        orf_mapper.to_csv('../permutations/orf_permutation.csv')
        mapper.to_csv('../permutations/gene_permutation.csv')

    return mapper, crispr_mapper, orf_mapper


def get_modalities():
    '''Get dataset with modalities'''
    import os
    from utils_stanford import get_standard_experiments, get_raw_dataframe, get_median_consensus_profiles
    os.chdir('insert path here')
    experiments = get_standard_experiments()
    split = 'compound'
    filetype = 'normalized_feature_select_negcon.csv.gz'
    df = get_raw_dataframe(experiments, filetype)

    # Split the DataFrame into Compound and CRISPR/ORF DataFrames
    modality_dfs = {}  # modality_dfs['Compound'] contains the Compound df.
    for modality in experiments:
        modality_df = get_median_consensus_profiles(df, modality)
        modality_df.columns = ['{}_{}'.format(
            col, modality) for col in modality_df.columns]
        modality_dfs[modality] = modality_df
    return modality_dfs


def test_permutations():
    '''
    Test generation N times asserting it outputs always the same permutation
    '''
    gene_mapper, orf_mapper, crispr_mapper = generate_permutations()
    for _ in range(10):
        new_gene, new_orf, new_crispr = generate_permutations()
        assert (orf_mapper == new_orf).all().all()
        assert (crispr_mapper == new_crispr).all().all()
        assert (gene_mapper == new_gene).all().all()
