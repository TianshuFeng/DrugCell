#%%
import os
os.getcwd()
#%%
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import MultivariateNormal

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import copy

import tqdm

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import xgboost as xgb

from codes.utils.util import *
from codes.drugcell_NN import *

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'
  
torch.cuda.device_count()

#%%
DEVICE = 'cuda:6'

#%% Data Loading
def set_col_names_in_multilevel_dataframe(
    df: pd.DataFrame,
    level_map: dict,
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol") -> pd.DataFrame:
    """ Util function that supports loading of the omic data files.
    Returns the input dataframe with the multi-level column names renamed as
    specified by the gene_system_identifier arg.

    Args:
        df (pd.DataFrame): omics dataframe
        level_map (dict): encodes the column level and the corresponding identifier systems
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: the input dataframe with the specified multi-level column names
    """
    df = df.copy()

    level_names = list(level_map.keys())
    level_values = list(level_map.values())
    n_levels = len(level_names)
    
    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    # print(gene_system_identifier)
    # import pdb; pdb.set_trace()
    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            df.columns = df.columns.rename(level_names, level=level_values)  # assign multi-level col names
        else:
            df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retian specific column level
    else:
        assert len(gene_system_identifier) <= n_levels, f"'gene_system_identifier' can't contain more than {n_levels} items."
        set_diff = list(set(gene_system_identifier).difference(set(level_names)))
        assert len(set_diff) == 0, f"Passed unknown gene identifiers: {set_diff}"
        kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
        # print(list(kk.keys()))
        # print(list(kk.values()))
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
        drop_levels = list(set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)
    return df


def load_gene_expression_data(gene_expression_file_path, 
    gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
    sep: str="\t",
    verbose: bool=True) -> pd.DataFrame:
    """
    Returns gene expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(gene_expression_file_path, sep=sep, index_col=0, header=header)

    df.index.name = "improve_sample_id"  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Gene expression data: {df.shape}")
    return df

gene_express = load_gene_expression_data("data/GDSC/cancer_gene_expression.tsv")
response = pd.read_csv("data/GDSC/response.tsv", sep = '\t')
gdsc_info = pd.read_csv("data/GDSC/GDSC_info.csv")
gdsc = pd.merge(gene_express, gdsc_info, how="inner", left_on=["improve_sample_id"], right_on=['ModelID'])
gdsc_x = gdsc.loc[:, gene_express.columns]
gdsc_y = gdsc.loc[:, 'DepmapModelType']

#%% Merging with ontology
training_file = "data/drugcell_train.txt"
testing_file = "data/drugcell_test.txt"
val_file = "data/drugcell_val.txt"
cell2id_file = "data/cell2ind.txt"
drug2id_file = "data/drug2ind.txt"
genotype_file = "data/cell2mutation.txt"
fingerprint_file = "data/drug2fingerprint.txt"
onto_file = "data/drugcell_ont.txt"
gene2id_file = "data/gene2ind.txt"

train_data, feature_dict, cell2id_mapping, drug2id_mapping = prepare_train_data(training_file, 
                                                                  testing_file, cell2id_file, 
                                                                  drug2id_file)

gene2id_mapping = load_mapping(gene2id_file)

# load cell/drug features
cell_features = np.genfromtxt(genotype_file, delimiter=',')
drug_features = np.genfromtxt(fingerprint_file, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

# load ontology
dG, root, term_size_map, \
    term_direct_gene_map = load_ontology(onto_file, 
                                         gene2id_mapping)
    
gene_intersect_list = list(set(gene2id_mapping.keys()) & set(gdsc_x.columns))
gdsc_tensor = torch.zeros(gdsc_x.shape[0], num_genes)

for gene in gene_intersect_list:
    idx = gene2id_mapping[gene]
    gdsc_tensor[:,idx] = torch.tensor(gdsc_x[gene])

cancer_2_idx = {}
idx_2_cancer = {}
cancer_type_idx = []

i = 0
for cancer in gdsc_y:
    if cancer not in cancer_2_idx:
        cancer_2_idx[cancer] = i
        idx_2_cancer[i] = cancer
        cancer_type_idx.append(i)
        
        i += 1
    else:
        cancer_type_idx.append(cancer_2_idx[cancer])

#%% Model training
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10)

X = np.array(gdsc_tensor)
y = np.array(cancer_type_idx)

X_train = X[:700,:]
y_train = y[:700]

X_test = X[700:,:]
y_test = y[700:]

#%%
rf_res = rf.fit(X_train, y_train)
# boosting_res = boosting.fit(X_train, y_train)
# %%
print(np.mean(rf_res.predict(X_train) == y_train))
print(np.mean(rf_res.predict(X_test) == y_test))
# %%

bst = xgb.XGBClassifier(n_estimators=10, 
                        max_depth=4, n_jobs=4, 
                        objective='multi:softmax',
                        reg_lambda = 10,
                        reg_alpha = 2.5)
res_bst = bst.fit(X_train, y_train)

# %%
print(np.mean(res_bst.predict(X_train) == y_train))
print(np.mean(res_bst.predict(X_test) == y_test))
# %% LUAD or not
y_luad = copy.deepcopy(y)
y_luad[y == 2] =1
y_luad[y != 2] = 0

y_luad_train = y_luad[:700]
y_luad_test = y_luad[700:]

# %%
bst = xgb.XGBClassifier(n_estimators=10, max_depth=4, 
                        n_jobs=4, objective='binary:logistic',
                        eval_metric = 'auc')
res_bst = bst.fit(X_train, y_luad_train, eval_set = [(X_test, y_luad_test)])

# %%
print(np.mean(res_bst.predict(X_test) == y_luad_test))
# %%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_luad_test, res_bst.predict(X_test, output_margin = True))

# %%
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_luad_test,
    res_bst.predict(X_test, output_margin = True),
)
plt.axis("square")
plt.legend()
plt.show()
# %%  Top 5 vs rest

y_top5 = copy.deepcopy(y)

for idx, y_tmp in enumerate(y):
    if y_tmp in [2,13,24,30,14]:
        y_top5[idx] = 1
    else:
        y_top5[idx] = 0
    
y_top5_train = y_top5[:700]
y_top5_test = y_top5[700:]
# %%
bst = xgb.XGBClassifier(n_estimators=20, max_depth=4, 
                        n_jobs=4, objective='binary:logistic',
                        eval_metric = 'auc')
res_bst = bst.fit(X_train, y_top5_train, eval_set = [(X_test, y_top5_test)])

# %%
