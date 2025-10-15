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
class dcell_nn(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, root, 
                 num_hiddens_genotype, num_hiddens_final):

        super(dcell_nn, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        self.term_visit_count = {}
        self.init_term_visits(term_size_map)
        
        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.term_dim_map = {}
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        self.gene_dim = ngene

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for final layer TODO: modify it into VAE
        final_input_size = num_hiddens_genotype # + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final * 2))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final * 2))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final * 2,1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))


    def init_term_visits(self, term_size_map):
        
        for term in term_size_map:
            self.term_visit_count[term] = 0
    
    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
#            print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output


    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
            self.add_module(term+'_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []   # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            #leaves = [n for n,d in dG.out_degree().items() if d==0]
            #leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden,1))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(1,1))

            dG.remove_nodes_from(leaves)


    # definition of forward function
    def forward(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                self.term_visit_count[term] += 1
                
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list,1)

                term_NN_out = self._modules[term+'_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)

        # connect two neural networks at the top #################################################
        final_input = term_NN_out_map[self.root] # torch.cat((term_NN_out_map[self.root], drug_out), 1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map

# %%

training_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drugcell_train.txt"
testing_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drugcell_test.txt"
val_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drugcell_val.txt"
cell2id_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/cell2ind.txt"
drug2id_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drug2ind.txt"
genotype_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/cell2mutation.txt"
fingerprint_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drug2fingerprint.txt"
onto_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/drugcell_ont.txt"
gene2id_file = "/Users/tianshu/Library/CloudStorage/Dropbox/Research/#Topics/Mayo_related/Codes/DrugCell/data/gene2ind.txt"

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

# load the number of hiddens #######
num_hiddens_genotype = 6

# num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = 6

#%%

h = [v for v in term_size_map.values()]
plt.hist(h, bins=30, range=(100, 2000))


# %%

model = dcell_nn(term_size_map, term_direct_gene_map, dG, num_genes, 
                 root, num_hiddens_genotype, num_hiddens_final)

train_feature, train_label, test_feature, test_label = train_data

# %%
train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=64, shuffle=False)
test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=64, shuffle=False)

(inputdata, labels) = next(iter(train_loader))
# %%
features = build_input_vector(inputdata, cell_features, drug_features)
# %%
aux_out_map, term_NN_out_map = model(features)
# %%
def create_term_mask(term_direct_gene_map, gene_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), gene_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = torch.autograd.Variable(mask)

		term_mask_map[term] = mask_gpu

	return term_mask_map

term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes)

#%%
h = [v for v in model.term_visit_count.values()]
plt.hist(h)

# %%
gene_input = features.narrow(1, 0, model.gene_dim)

# define forward function for genotype dcell #############################################
term_gene_out_map = {}

for term, _ in model.term_direct_gene_map.items():
    term_gene_out_map[term] = model._modules[term + '_direct_gene_layer'](gene_input)

term_NN_out_map = {}
aux_out_map = {}

for i, layer in enumerate(model.term_layer_list):

    for term in layer:

        child_input_list = []

        model.term_visit_count[term] += 1
        
        for child in model.term_neighbor_map[term]:
            child_input_list.append(term_NN_out_map[child])

        if term in model.term_direct_gene_map:
            child_input_list.append(term_gene_out_map[term])

        child_input = torch.cat(child_input_list,1)

        term_NN_out = model._modules[term+'_linear_layer'](child_input)

        Tanh_out = torch.tanh(term_NN_out)
        term_NN_out_map[term] = model._modules[term+'_batchnorm_layer'](Tanh_out)
        aux_layer1_out = torch.tanh(model._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
        aux_out_map[term] = model._modules[term+'_aux_linear_layer2'](aux_layer1_out)
    break
# %%
