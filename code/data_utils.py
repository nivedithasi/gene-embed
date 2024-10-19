import os
import glob
import pandas as pd

def load_data(exp, plate, filetype):
    """load all data from a single experiment into a single dataframe"""
    path = os.path.join('../pilot-cpjump1-data/profiles',
                        f'*{exp}',
                        f'{plate}',
                        f'*_{filetype}')
    files = glob.glob(path)
    print(f"Files to read: {files}")
    df = pd.concat(pd.read_csv(_, low_memory=False) for _ in files)
    return df

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]

def get_featurecols(df):
    """return a list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]

def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]

def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]

def remove_edge_compounds(df):
    """return dataframe of just inner wells"""
    edge_compounds = list(df[df['Metadata_Well'].str.contains('^[AP]|01$|24$')]['Metadata_pert_id'].unique())
    edge_compounds = [edge_compounds[i] for i in range(len(edge_compounds))]
    df = df[~df['Metadata_pert_id'].isin(edge_compounds)].reset_index(drop=True)
    return df

def remove_negcon_empty_wells(df):
    """return dataframe of non-negative control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=['Metadata_broad_sample'])
        .reset_index(drop=True)
    )
    return df

def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = (
        df.dropna(subset=['Metadata_broad_sample'])
        .reset_index(drop=True)
    )
    return df