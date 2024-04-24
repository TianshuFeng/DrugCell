# import anndata
import pandas as pd
from typing import List, Union
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import MultivariateNormal

import os
import copy

import tqdm
from sklearn.metrics import roc_auc_score

class Drugcell_Vae(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, 
                 num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, 
                 n_class, inter_loss_penalty = 0.2):

        super().__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug
        
        
        self.num_hiddens_final = num_hiddens_final
        self.n_class = n_class
        self.inter_loss_penalty = inter_loss_penalty
        self.dG = copy.deepcopy(dG)

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        self.term_visit_count = {}
        self.init_term_visits(term_size_map)
        
        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.term_dim_map = {}
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        self.gene_dim = ngene
        self.drug_dim = ndrug

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(self.dG)

        # add modules for neural networks to process drugs
        self.construct_NN_drug()

        # add modules for final layer TODO: modify it into VAE
        final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final * 2))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final * 2))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final * 2, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))
        
        self.decoder_affine = nn.Linear(num_hiddens_final, 1)

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


    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        for i in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(i+1), nn.Linear(input_size, self.num_hiddens_drug[i]))
            self.add_module('drug_batchnorm_layer_' + str(i+1), nn.BatchNorm1d(self.num_hiddens_drug[i]))
            self.add_module('drug_aux_linear_layer1_' + str(i+1), nn.Linear(self.num_hiddens_drug[i],1))
            self.add_module('drug_aux_linear_layer2_' + str(i+1), nn.Linear(1,1))

            input_size = self.num_hiddens_drug[i]

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
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden, term_hidden))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(term_hidden, 1))

            dG.remove_nodes_from(leaves)


    # definition of encoder
    def encoder(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)
        
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

        drug_out = drug_input

        for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)]( torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_'+str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules['drug_aux_linear_layer1_'+str(i)](drug_out))
            aux_out_map['drug_'+str(i)] = self._modules['drug_aux_linear_layer2_'+str(i)](aux_layer1_out)


        # connect two neural networks at the top #################################################
        final_input = torch.cat((term_NN_out_map[self.root], drug_out), 1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, term_NN_out_map
    
    def forward(self, x):
        
        aux_out_map, term_NN_out_map = self.encoder(x)
        
        mu = term_NN_out_map['final'][..., :self.num_hiddens_final]
        log_var = term_NN_out_map['final'][..., :self.num_hiddens_final]  # T X batch X z_dim
        std_dec = log_var.mul(0.5).exp_()
        # std_dec = 1
        
        latent = MultivariateNormal(loc = mu, 
                                    scale_tril=torch.diag_embed(std_dec))
        z = latent.rsample()
        
        recon_mean = self.decoder_affine(z)
        recon_mean = F.sigmoid(recon_mean)

        return recon_mean, mu, log_var, aux_out_map, term_NN_out_map
    
    def loss_log_vae(self, recon_mean, y, mu, log_var, beta = 0.001):
        # y: true labels
        ori_y_shape = y.shape
        
        class_loss = F.mse_loss(recon_mean.view(-1), 
                                     y.reshape(-1), reduction = 'none').div(np.log(2)).view(*ori_y_shape)
        
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 
                              dim = -1)
        
        log_loss = class_loss + beta * KLD
        log_loss = torch.mean(torch.logsumexp(log_loss, 0))
        
        return log_loss
    
    def intermediate_loss(self, aux_out_map, y):
        
        inter_loss = 0
        for name, output in aux_out_map.items():
            if name == 'final':
                inter_loss += 0
            else: # change 0.2 to smaller one for big terms
                ori_y_shape = y.shape
        
                term_loss = F.mse_loss(output.view(-1), 
                                             y.reshape(-1), 
                                             reduction = 'none').div(np.log(2)).view(*ori_y_shape)
                inter_loss += term_loss

        return inter_loss
    

def create_term_mask(term_direct_gene_map, gene_dim, device):

    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask)

        term_mask_map[term] = mask_gpu.to(device)

    return term_mask_map

