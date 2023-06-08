#%%

import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import util
from utils.util import *
from drugcell_NN import *
import argparse
import numpy as np
import time

import matplotlib.pyplot as plt

#%%

onto_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drugcell_ont.txt"
gene2id_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/gene2ind.txt"

gene2id_mapping = load_mapping(gene2id_file)

dG, root, term_size_map, \
    term_direct_gene_map = load_ontology(onto_file, 
                                         gene2id_mapping)

# %%
