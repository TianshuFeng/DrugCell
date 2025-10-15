import sys
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class Drugcell_Vae(nn.Module):
    
    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, root, 
                 num_hiddens_genotype, num_hiddens_final, n_class, inter_loss_penalty = 0.001,
                 turn_off_variational=False):

        super(Drugcell_Vae, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_final = num_hiddens_final
        self.n_class = n_class
        self.inter_loss_penalty = inter_loss_penalty
        self.dG = copy.deepcopy(dG)
        
        self.turn_off_variational = turn_off_variational  # Whether to turn VAE into AE

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
        self.construct_NN_graph(self.dG)

        # add modules for final layer TODO: modify it into VAE
        final_input_size = num_hiddens_genotype # + num_hiddens_drug[-1]
        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final * 2))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final * 2))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final * 2, num_hiddens_final * 2))
        self.add_module('final_linear_layer_output', nn.Linear(num_hiddens_final * 2, final_input_size))
        
        self.decoder_affine = nn.Linear(num_hiddens_final, ngene)

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
        self.term_list = {}
        term_index = 0

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

            gene_list = []

            for term in leaves:
                # change1
                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0
                gene_size = 0

                self.term_list[term] = term_index

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])
                    gene_size = len(self.term_direct_gene_map[term])


                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]


                term_index += 1

            
                self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden, term_hidden))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(term_hidden, input_size))
                self.add_module(term+'_aux_linear_layer3', nn.Linear(term_hidden, 32))
                self.add_module(term+'_aux_linear_layer4', nn.Linear(32, self.n_class))
                
                self.add_module('discriminator1' + term, nn.Linear(input_size,2))
                self.add_module('discriminator2' + term, nn.Linear(2,1))

            dG.remove_nodes_from(leaves)
    
        return self.term_layer_list


    # definition of encoder
    def encoder(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, 0)
        
        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        term_NN_out_map = {}
        aux_out_map = {}
        aux_layer1_out = {}
        term_original_children = {}
        aux_cancer_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                term_hidden = self.term_dim_map[term]

                child_input_list = []
                gene_input = []

                self.term_visit_count[term] += 1
                
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list,1)
                
                term_original_children[term] = child_input
                
                term_NN_out = self._modules[term+'_linear_layer'](child_input)
                """term_NN_out = self.dropout(term_NN_out)"""

                tanh = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](tanh)
                """term_NN_out_map[term] = self.dropout(term_NN_out_map[term])"""

                aux_layer1_out[term] = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
                """aux_layer1_out =  self.dropout(aux_layer1_out)"""
                
                aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out[term])

                aux_layer3_out = torch.tanh(self._modules[term+'_aux_linear_layer3'](term_NN_out_map[term]))

                aux_cancer_map[term] = self._modules[term+'_aux_linear_layer4'](aux_layer3_out)

        drug_out = drug_input

        """for i in range(1, len(self.num_hiddens_drug)+1, 1):
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)](torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_'+str(i)] = drug_out"""



        # connect two neural networks at the top #################################################
        final_input = torch.tanh(term_NN_out_map[self.root])

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        return aux_out_map, aux_cancer_map, term_NN_out_map, term_original_children


    
    def forward(self, x):
        
        aux_out_map, aux_cancer_map, term_NN_out_map, term_original_children = self.encoder(x)
        
        mu = term_NN_out_map['final'][..., :self.num_hiddens_final]
        log_var = term_NN_out_map['final'][..., -self.num_hiddens_final:] # T X batch X z_dim
        std_dec = log_var.mul(0.5).exp_()
        # std_dec = 1
        
        if not self.turn_off_variational:
            latent = MultivariateNormal(loc = mu, 
                                        scale_tril=torch.diag_embed(std_dec))
            z = latent.rsample()
            
            recon_mean = self.decoder_affine(z)
            logits = F.softmax(recon_mean, -1)
        else:
            recon_mean = self.decoder_affine(mu)
            logits = F.softmax(recon_mean, -1)

        return logits, mu, log_var, aux_out_map, aux_cancer_map, term_NN_out_map, term_original_children

        
    

    def loss_log_vae(self, recon_mean, y, mu, log_var, beta = 0.001):
            # y: true labels

        ori_y_shape = y.shape
        
        class_loss = F.mse_loss(recon_mean.view(-1), 
                                     y.reshape(-1), reduction = 'none').div(np.log(2)).view(*ori_y_shape)
        
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 
                              dim = -1)
        
        class_loss = class_loss.mean(dim=-1)

        log_loss = class_loss + beta * KLD
        log_loss = torch.mean(torch.logsumexp(log_loss, 0))
        
        return log_loss


    def intermediate_loss_cancer(self, aux_out_map, y):
        
        inter_loss = 0
        for name, output in aux_out_map.items():
            if name == 'final':
                inter_loss += 0
            else: # change 0.2 to smaller one for big terms
                ori_y_shape = y.shape
        
                term_loss = F.cross_entropy(output.view(-1, output.shape[-1]), 
                                             y.reshape(-1), 
                                             reduction = 'none').div(np.log(2)).view(*ori_y_shape)
                inter_loss += term_loss

        return inter_loss
        
    
    
    def intermediate_loss(self, aux_out_map, term_original_children):
        inter_loss = 0
        for name, output in term_original_children.items():
            if name == 'final':
                inter_loss += 0
            else:
                
                """out_prob = F.softmax(output)
                term_loss = F.cross_entropy(aux_out_map[name],
                                            out_prob, 
                                            reduction = 'none')"""
                
                

                #   MSE loss
                mse_loss = F.mse_loss(output, aux_out_map[name], reduction = 'none')
                term_loss = torch.sum(mse_loss)

                inter_loss += term_loss


        return inter_loss
    
    def sparse_loss(self, recon_mean, sparsity_target, sparsity_weight):
    
        rho_hat = torch.mean(recon_mean, dim=0) 
        epsilon = 1e-6
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)

        kl_div = sparsity_target * torch.log(sparsity_target / rho_hat) + \
             (1 - sparsity_target) * torch.log((1 - sparsity_target) / (1 - rho_hat))
             
        return sparsity_weight * torch.sum(kl_div)

    def contrastive_loss_with_sparsity(self, aux_out_map, term_original_children, sparsity_target, sparsity_weight):
        inter_loss = 0

        for name, output in term_original_children.items():
            if name == 'final':
                inter_loss += 0
            else:
                inter_loss += self.sparse_loss(aux_out_map[name], sparsity_target = sparsity_target, sparsity_weight = sparsity_weight)

        return inter_loss

    def mask_input(self, input_tensor, mask_ratio):

        mask = torch.rand_like(input_tensor) < mask_ratio
        masked_input = input_tensor.clone()
        masked_input[mask] = 0 
        return masked_input, mask

