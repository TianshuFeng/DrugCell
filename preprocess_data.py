import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import MultivariateNormal
from codes.utils.util import *
import argparse
import candle
import copy
import tqdm
from pathlib import Path
import logging
import sys
import pandas as pd
import sklearn
import os
import torch.optim as optim
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
import time
from time import time

from sklearn.metrics import roc_auc_score


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

file_path = os.path.dirname(os.path.realpath(__file__))
fdir = Path('__file__').resolve().parent


required = None
additional_definitions = None

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/tianshu/DrugCell/hpo_data/'
data_dir = str(fdir) + '/hpo_data/'
print(data_dir)

# additional definitions
additional_definitions = [
    {
        "name": "num_hiddens_genotype",
        "type": int,
        "help": "total number of hidden genotypes",
    },
    {
        "name": "num_hiddens_final",
        "type": int,
        "help": "number of hideen final",
    },
    {   
        "name": "drug_hiddens",
        "type": str,
        "help": "list of hidden drugs",
    },
    {
        "name": "learning_rate",
        "type": float,
        "help": "learning rate of the model",
    },
    {   
        "name": "betas_adam",
        "type": str, 
        "help": "tuple of values ",
    },
    {   
        "name": "cuda",
        "type": int, 
        "help": "CUDA ID",
    },
    {   
        "name": "eps_adam",
        "type": float, 
        "help": "episilon of the optimizer",
    },
    {   
        "name": "direct_gene_weight_param",
        "type": int, 
        "help": "weight of the genes",
    },
    {   
        "name": "batch_size",
        "type": int, 
        "help": "batch size for data processing",
    },
    {   
        "name": "beta_kl",
        "type": float, 
        "help": "KL divergenece beta",
    },
    {   
        "name": "optimizer",
        "type": str, 
        "help": "type of optimerze",
    },        
    {  
        "name": "epochs",
        "type": int, 
        "help": "total number of epochs",
    },
]

# required definitions
required = [
    "genotype",
    "fingerprint",
]


class DrugCell_candle(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions

def initialize_parameters():
    preprocessor_bmk = DrugCell_candle(file_path,
        'DrugCell_params.txt',
        'pytorch',
        prog='DrugCell',
        desc='tianshu drugcell'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters


def load_params(params, data_dir):
    print(os.environ['CANDLE_DATA_DIR'])
    args = candle.ArgumentStruct(**params)
    drug_tensor_path = data_dir + params['drug_tensor']
    params['drug_tensor'] = drug_tensor_path
    data_tensor_path = data_dir + params['data_tensor']
    params['data_tensor'] = data_tensor_path
    response_data_path = data_dir + params['response_data']
    params['response_data'] = response_data_path
    train_data_path = data_dir + params['train_data']
    params['train_data'] = train_data_path    
    test_data_path = data_dir + params['test_data']
    params['test_data'] = test_data_path
    val_data_path = data_dir + params['val_data']
    params['val_data'] = val_data_path
    onto_data_path = data_dir + params['onto_file']
    params['onto'] = onto_data_path
    cell2id_path = data_dir + params['cell2id'] 
    params['cell2id'] = cell2id_path
    drug2id_path  = data_dir + params['drug2id']
    params['drug2id'] = drug2id_path
    gene2id_path = data_dir + params['gene2id']
    params['gene2id'] = gene2id_path
    genotype_path = data_dir + params['genotype']
    params['genotype'] = genotype_path
    fingerprint_path = data_dir + params['fingerprint']
    params['fingerprint'] = fingerprint_path
    hidden_path = data_dir + params['hidden']
    params['hidden_path'] = hidden_path
    output_dir_path = data_dir + params['output_dir']
    params['output_dir'] = output_dir_path
    params['result'] = output_dir_path
    return(params)

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


def preprocess_data(params):
    response_gdcs2 = torch.tensor(np.loadtxt(params['response_data'],delimiter=",", dtype=np.float32))
    gdsc_tensor = torch.tensor(np.loadtxt(params['data_tensor'],
                                          delimiter=",", dtype=np.float32))
    drug_tensor = torch.tensor(np.loadtxt(params['drug_tensor'],
                                          delimiter=",", dtype=np.float32))
    num_drugs = drug_tensor.shape[1]
    train_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[:423]
    test_gdcs_idx = torch.unique(response_gdcs2[:,0], sorted=False)[423:]

    gdsc_data = GDSCData(response_gdcs2, gdsc_tensor, drug_tensor)
    #gdsc_data_train = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], train_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    gdsc_data_train = GDSCData(response_gdcs2[np.isin(response_gdcs2[:,0], train_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    #gdsc_data_test = GDSCData(response_gdcs2[torch.isin(response_gdcs2[:,0], test_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    gdsc_data_test = GDSCData(response_gdcs2[np.isin(response_gdcs2[:,0], test_gdcs_idx)].float(), gdsc_tensor, drug_tensor)
    return num_drugs, gdsc_data_train, gdsc_data_test


def process_drugcell_inputs(params):
    training_file = params['train_data']
    testing_file = params['test_data']
    val_file = params['val_data']
    cell2id_file = params['cell2id']
    drug2id_file = params['drug2id']
    genotype_file = params['genotype']
    fingerprint_file = params['fingerprint']
    onto_file = params['onto']
    gene2id_file = params['gene2id']
    
    train_data, feature_dict, cell2id_mapping, drug2id_mapping = prepare_train_data(training_file, 
                                                                                    testing_file, cell2id_file, 
                                                                                    drug2id_file)
   
    gene2id_mapping = load_mapping(gene2id_file)
    cell_features = np.genfromtxt(genotype_file, delimiter=',')
    drug_features = np.genfromtxt(fingerprint_file, delimiter=',')
    num_genes = len(gene2id_mapping)
#    # load ontology
    dG, root, term_size_map,term_direct_gene_map = load_ontology(onto_file,gene2id_mapping)
    return dG, root, term_size_map,term_direct_gene_map, num_genes


def run(params):
    num_drugs, gdsc_data_train, gdsc_data_test = preprocess_data(params)
    train_file  = params['train_ml_data_dir'] + "/train_data.pt"
    test_file  = params['train_ml_data_dir'] + "/test_data.pt"
    torch.save(gdsc_data_train,train_file)
    torch.save(gdsc_data_train,test_file)
    print(num_drugs)
    
def candle_main():
    params = initialize_parameters()
    params = load_params(params, data_dir)
    run(params)

if __name__ == "__main__":
    candle_main()
