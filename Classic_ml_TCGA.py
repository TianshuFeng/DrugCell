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
DEVICE = 'cuda:7'

#%% Data Loading
class RNASeqData(Dataset):
    
    def __init__(self, X, c=None, y=None, transform=None):
        self.X = X
        self.y = y
        self.c = c
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        sample = self.X[index,:]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.y is not None and self.c is None:
            return sample, self.y[index]
        elif self.y is not None and self.c is not None:
            return sample, self.y[index], self.c[index]
        elif self.y is None and self.c is not None:
            return sample, self.c[index]
        else:
            return sample

rna_seq = pd.read_csv('data/tcga/train_tcga_expression_matrix_processed.tsv',sep='\t',index_col=0)
tcga_mad_genes = pd.read_csv('data/tcga/tcga_mad_genes.tsv', sep='\t')
tcga_sample_counts = pd.read_csv('data/tcga/tcga_sample_counts.tsv', sep='\t')
rna_seq = rna_seq / rna_seq.std()
rna_seq = np.log(rna_seq + 1)
tcga_sample_identifiers = pd.read_csv('data/tcga/tcga_sample_identifiers.tsv', sep='\t',index_col=0)
gene_id_dict = pd.read_csv('data/tcga/gene_dict.csv')

tcga_df = rna_seq

new_column = []
for col_name in tcga_df.columns:
    col_loc = (gene_id_dict.entrezgene_id == int(col_name))
    if np.sum(col_loc) == 0:
        tcga_df = tcga_df.drop(columns = col_name)
    else:
        new_column.append(gene_id_dict.hgnc_symbol[gene_id_dict.entrezgene_id == int(col_name)].iloc[0])
        


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
    
tcga_gene2id = {}
for idx, gene in enumerate(tcga_df.columns):
    tcga_gene2id[gene] = idx

gene_intersect_list = list(set(gene2id_mapping.keys()) & set(tcga_df.columns))
tcga_tensor = torch.zeros(tcga_df.shape[0], num_genes)
cancer_type = tcga_sample_identifiers.loc[tcga_df.index, 'cancer_type']

cancer_2_idx = {}
idx_2_cancer = {}
cancer_type_idx = []

i = 0
for cancer in cancer_type:
    if cancer not in cancer_2_idx:
        cancer_2_idx[cancer] = i
        idx_2_cancer[i] = cancer
        cancer_type_idx.append(i)
        
        i += 1
    else:
        cancer_type_idx.append(cancer_2_idx[cancer])


#%%
torch.manual_seed(0)
y = torch.tensor(cancer_type_idx)
tcga_dataset = RNASeqData(X = tcga_tensor, y = y)
training_set, testing_set = random_split(tcga_dataset, [0.7, 0.3])

#%%

train_loader = DataLoader(training_set, batch_size=len(training_set), shuffle=False)
test_loader = DataLoader(testing_set, batch_size=len(testing_set), shuffle=True)
x_train_tensor, y_train_tensor = next(iter(train_loader))
x_test_tensor, y_test_tensor = next(iter(test_loader))

#%% Model training
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10)


X_train = np.array(x_train_tensor)
y_train = np.array(y_train_tensor)

X_test = np.array(x_test_tensor)
y_test = np.array(y_test_tensor)

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

# %%

# %% With Drug embedding
drug_info_data = pd.read_csv('data/GDSC/drug_info.tsv', sep='\t')
response = pd.read_csv("data/GDSC/response.tsv", sep = '\t')
response_gdcs2 = response[response['source'] == 'GDSCv2'].loc[:,['improve_sample_id', 
                                                             'improve_chem_id',
                                                         'auc']]

# %%
a = response_gdcs2['improve_chem_id'].value_counts()
# %%
GDSC_drug = drug_info_data[drug_info_data['improve_chem_id'].isin(a.index)]
GDSC_drug = GDSC_drug.loc[GDSC_drug['DrugID'].str.startswith('GDSC')]
# %% With Drug embedding

drug_info_data = pd.read_csv('data/GDSC/drug_info.tsv', sep='\t')
response = pd.read_csv("data/GDSC/response.tsv", sep = '\t')
response_gdcs2 = response[response['source'] == 'GDSCv2'].loc[:,['improve_sample_id', 
                                                             'improve_chem_id',
                                                         'auc']]
gene_express = load_gene_expression_data("data/GDSC/cancer_gene_expression.tsv")
drug_ecfp4_nbits512 = pd.read_csv("data/GDSC/drug_ecfp4_nbits512.tsv", sep = '\t', index_col=0)

gdsc_row_key_id = {k: v for v, k in enumerate(gene_express.index)}
gdsc_row_id_key = {v: k for v, k in enumerate(gene_express.index)}

chem_row_key_id = {k: v for v, k in enumerate(drug_ecfp4_nbits512.index)}
chem_row_id_key = {v: k for v, k in enumerate(drug_ecfp4_nbits512.index)}
# %%
response_gdcs2 = response_gdcs2.replace({'improve_sample_id': gdsc_row_key_id})
response_gdcs2 = response_gdcs2.replace({'improve_chem_id': chem_row_key_id})
response_gdcs2 = torch.tensor(response_gdcs2.to_numpy())

train_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[:423]
test_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[423:]

drug_tensor = torch.tensor(drug_ecfp4_nbits512.to_numpy())
#%%
class GDSCData(Dataset):
    
    def __init__(self, response, gene_tensor, chem_tensor):
        self.response = response
        self.gene_tensor = gene_tensor
        self.chem_tensor = chem_tensor
        
    def __len__(self):
        return self.response.shape[0]
    
    def __getitem__(self, index):
        sample = self.response[index,:]
        
        X_gene = self.gene_tensor[sample[0].long() ,:]
        X_chem = self.chem_tensor[sample[1].long() ,:]
        
        y = sample[2]

        X = torch.cat((X_gene, X_chem), 0)
        
        return X, y

gdsc_data = GDSCData(response_gdcs2, gdsc_tensor, drug_tensor)
gdsc_data_train = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], train_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
gdsc_data_test = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], test_gdcs_idx)].float(), gdsc_tensor, drug_tensor)

train_loader = DataLoader(gdsc_data_train,
                          batch_size=len(gdsc_data_train), shuffle=False)
test_loader = DataLoader(gdsc_data_test, 
                         batch_size=len(gdsc_data_test), shuffle=False)

(train_data, train_response) = next(iter(train_loader))
(test_data, test_response) = next(iter(test_loader))


#%%
bst_drug = xgb.XGBRegressor(n_estimators=50, max_depth=4, 
                        n_jobs=4, objective='reg:squarederror',
                        eval_metric = 'rmse')

res_bst_drug = bst_drug.fit(train_data.numpy(), train_response.numpy(), 
                  eval_set = [(test_data.numpy(), test_response.numpy())])

# RMSE 0.106

#%%
from sklearn.ensemble import RandomForestRegressor

rf_drug = RandomForestRegressor(n_estimators = 10, n_jobs=4, verbose=1)

rf_res_drug = rf_drug.fit(train_data.numpy(), train_response.numpy())

# %%
rf_predict_drug = rf_drug.predict(test_data.numpy())

rf_rmse = np.sqrt( np.mean((rf_predict_drug - test_response.numpy())**2) )

# 0.099
# %%
from joblib import dump, load
dump(rf_res_drug, 'rf_res_drug.joblib') 
# %%
