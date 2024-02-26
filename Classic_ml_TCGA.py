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
from sklearn.metrics import adjusted_rand_score, average_precision_score

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

#region Loading Model and data loading
#%% #tag Data Loading
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

#%%
tcga_df.columns = new_column

#%% #tag Merging with ontology
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

for gene in gene_intersect_list:
    idx = gene2id_mapping[gene]
    tcga_tensor[:,idx] = torch.tensor(tcga_df[gene])

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

#endregion
#region Classic_model 
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

np.random.seed(0)

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

tmp_aps = 0
for label in np.unique(y_test):
    tmp_aps += np.sum(y_test == label) * average_precision_score(y_true=y_test == label, 
                                                                 y_score=rf_res.predict(X_test) == label)
print(tmp_aps/len(y_test))  # 0.779

#%%
# cv_rf = cross_validate(rf, X_train, y_train, cv=5,scoring='accuracy', n_jobs = 2)
#%%
# print(np.std(cv_rf['test_score']))
#0.01009741191670707
# %%

bst = xgb.XGBClassifier(n_estimators=5, 
                        max_depth=4, n_jobs=4, 
                        objective='multi:softmax',
                        eval_metric = "merror")
res_bst = bst.fit(X_train, y_train,
                  eval_set = [(X_test, y_test)])

# %%
print(np.mean(res_bst.predict(X_train) == y_train))
print(np.mean(res_bst.predict(X_test) == y_test))

tmp_aps = 0
for label in np.unique(y_test):
    tmp_aps += np.sum(y_test == label) * average_precision_score(y_true=y_test == label, 
                                                                 y_score=res_bst.predict(X_test) == label)
print(tmp_aps/len(y_test))  # 0.810

#%%
# cv_bst = cross_validate(bst, X_train, y_train, cv=5,scoring='accuracy', n_jobs = 5)
#%%
# print(np.std(cv_bst['test_score']))  # 0.00979799421574557

# %%
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(hidden_layer_sizes=[16], random_state=1, max_iter=300)
mlp_clf.fit(X_train, y_train)
#%%
print(np.mean(mlp_clf.predict(X_train) == y_train))
print(np.mean(mlp_clf.predict(X_test) == y_test))

tmp_aps = 0
for label in np.unique(y_test):
    tmp_aps += np.sum(y_test == label) * average_precision_score(y_true=y_test == label, 
                                                                 y_score=mlp_clf.predict(X_test) == label)
print(tmp_aps/len(y_test))  # 0.888
# %%  Plot the results
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


cm = confusion_matrix(y_true=y_test, y_pred=rf_res.predict(X_test),
                      normalize='true')

# Plotting the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=False, fmt='.2g', cmap="Blues",
            xticklabels=list(cancer_2_idx.keys()),
            yticklabels=list(cancer_2_idx.keys()))
plt.plot(range(len(cancer_2_idx)+1), range(len(cancer_2_idx)+1), ls="-", lw=2, color="r")

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#endregion
# %%  Define DCell class

class dcell_vae(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, root, 
                 num_hiddens_genotype, num_hiddens_final, n_class, inter_loss_penalty = 0.2,
                 turn_off_variational=False):

        super(dcell_vae, self).__init__()

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
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final * 2, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))
        
        self.decoder_affine = nn.Linear(num_hiddens_final, n_class)

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
                self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden, self.n_class))
                self.add_module(term+'_aux_linear_layer2', nn.Linear(self.n_class, self.n_class))

            dG.remove_nodes_from(leaves)


    # definition of encoder
    def encoder(self, x):
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
    
    def forward(self, x):
        
        aux_out_map, term_NN_out_map = self.encoder(x)
        
        mu = term_NN_out_map['final'][..., :self.num_hiddens_final]
        log_var = term_NN_out_map['final'][..., :self.num_hiddens_final]  # T X batch X z_dim
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

        return logits, mu, log_var, aux_out_map, term_NN_out_map
    
    def loss_log_vae(self, logits, y, mu, log_var, beta = 0.001):
        # y: true labels
        ori_y_shape = y.shape
        
        class_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), 
                                     y.reshape(-1), reduction = 'none').div(np.log(2)).view(*ori_y_shape)
        if not self.turn_off_variational:
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 
                                dim = -1)
            log_loss = class_loss + beta * KLD
        else:
            log_loss = class_loss
        log_loss = torch.mean(torch.logsumexp(log_loss, 0))
        
        return log_loss
    
    def intermediate_loss(self, aux_out_map, y):
        
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

#region Plots For paper plots
#### %% For paper plots

test_loader = DataLoader(testing_set, batch_size=len(testing_set), shuffle=False)
(X_test_tensor, y_test_tensor) = next(iter(test_loader))

train_loader = DataLoader(training_set, batch_size=len(training_set), shuffle=False)
X_train_tensor, y_train_tensor = next(iter(train_loader))

dcell_model = torch.load("model_200_updated.pt", map_location=DEVICE)

#%%
np.random.seed(0)
torch.manual_seed(0)
logits, mu, log_var, aux_out_map, term_NN_out_map = dcell_model(X_test_tensor.to(DEVICE))


latent_param = term_NN_out_map['final'].detach().cpu()
mu = term_NN_out_map['final'][..., :dcell_model.num_hiddens_final].detach().cpu()
log_var = term_NN_out_map['final'][..., :dcell_model.num_hiddens_final].detach().cpu()

print(torch.sum(torch.argmax(logits, 1).cpu() == y_test_tensor)/len(y_test_tensor))

#%% 

#tag T-SNE Plot 

from sklearn.manifold import TSNE
import seaborn as sns

np.random.seed(0)

tsne = TSNE(n_components=2, perplexity=30)

# Fit the model to the data
tsne_tcga_data = tsne.fit_transform(mu.numpy())

y_test_tensor_str = [idx_2_cancer[idx.item()] for idx in y_test_tensor]

# Unique category labels: 'D', 'F', 'G', ...
color_labels = np.unique(y_test_tensor_str)

# List of RGB triplets
rgb_values = sns.color_palette("husl", len(color_labels))

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))

fig, ax = plt.subplots(1,1, figsize=(3, 2.5))
for category in color_labels:
    if category == 'COAD':
        mask = np.array(y_test_tensor_str) == category
        ax.scatter(tsne_tcga_data[mask, 0], tsne_tcga_data[mask, 1], s=10,
                color=color_map['COAD'], label='CRC')
    elif category == 'READ':
        continue
    else: 
        mask = np.array(y_test_tensor_str) == category
        ax.scatter(tsne_tcga_data[mask, 0], tsne_tcga_data[mask, 1], s=10,
                color=color_map[category], label=category)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(loc='lower left', bbox_to_anchor=(-0.5, -1.25), ncol=4)
plt.title('t-SNE projection of latent space for 32 cancer types')

plt.savefig('figure/TCGA_tsne.pdf', bbox_inches='tight', dpi=300)

#endregion


# %%
logit_list = []
for i in range(20):
    logits_tmp, _, _, _, _ = dcell_model(X_test_tensor.to(DEVICE))
    logit_list.append(logits_tmp.detach().cpu())

#%%
logit_stack = torch.stack(logit_list)
# logit_stack = torch.argmax(logit_stack, 2)
logit_stack = torch.mean(logit_stack, 0)
pred_vae = torch.argmax(logit_stack, 1)

print(torch.sum(pred_vae == y_test_tensor)/len(y_test_tensor))

# %%
with torch.no_grad():
    logits, mu_train, log_var, aux_out_map, term_NN_out_map = dcell_model(X_train_tensor.to(DEVICE))
    logits, mu_test, log_var, aux_out_map, term_NN_out_map = dcell_model(X_test_tensor.to(DEVICE))

#%%

mlp_clf = MLPClassifier(hidden_layer_sizes=[32], random_state=1, max_iter=300)
mlp_clf.fit(mu_train.detach().cpu(), y_train_tensor)
#%%
print(np.mean(mlp_clf.predict(mu_train.detach().cpu()) == y_train_tensor.numpy()))
print(np.mean(mlp_clf.predict(mu_test.detach().cpu()) == y_test_tensor.numpy()))

tmp_aps = 0
for label in np.unique(y_test_tensor.numpy()):
    tmp_aps += np.sum(y_test_tensor.numpy() == label) * average_precision_score(y_true=y_test_tensor.numpy() == label, 
                                                                 y_score=mlp_clf.predict(mu_test.detach().cpu()) == label)
print(tmp_aps/len(y_test_tensor.numpy()))

#%%
# res_list = []
# for i in range(10):
#     bootstrap_idx = torch.randint(mu_test.shape[0], (mu_test.shape[0],))
#     res_list.append(np.mean(mlp_clf.predict(mu_test.detach().cpu()[bootstrap_idx, :]) ==
#                             y_test_tensor[bootstrap_idx].numpy()))

#%% 


np.random.seed(0)

rf = RandomForestClassifier(n_estimators = 10)


X_train = np.array(x_train_tensor)
y_train = np.array(y_train_tensor)

X_test = np.array(x_test_tensor)
y_test = np.array(y_test_tensor)

#%%
rf_res = rf.fit(mu_train.detach().cpu(), y_train)
# boosting_res = boosting.fit(X_train, y_train)

#%%
from sklearn.model_selection import cross_validate
# cv_rf = cross_validate(rf, mu_train.detach().cpu(), y_train, cv=5,scoring='accuracy', n_jobs = 2)


# %%
print(np.mean(rf_res.predict(mu_train.detach().cpu()) == y_train))
print(np.mean(rf_res.predict(mu_test.detach().cpu()) == y_test))

# %%

bst = xgb.XGBClassifier(n_estimators=10, 
                        max_depth=4, n_jobs=4, 
                        objective='multi:softmax',
                        eval_metric = "merror")
res_bst = bst.fit(mu_train.detach().cpu(), y_train,
                  eval_set = [(mu_test.detach().cpu(), y_test)])

# %%
print(np.mean(res_bst.predict(mu_train.detach().cpu()) == y_train))
print(np.mean(res_bst.predict(mu_test.detach().cpu()) == y_test))

# %%

methods = ["VETE", "MLP", "XGBoost", "Random Forest"]
accu = [0.9225, 0.9277, 0.9125, 0.8754]
err_bar = [0.01, 0.01,0.01,0.01]

plt.figure(figsize = (3,2))
plt.bar(methods, accu)
plt.errorbar(methods, accu, yerr=err_bar, fmt=".", color="C3", capsize = 5.5)


plt.xlabel('Model Name')
plt.ylabel('Accuracy Score')
plt.ylim([0.7, 0.95])
plt.title('Model Performance Comparison\n(N=2,986)')
plt.tick_params(axis='x', rotation=45)


plt.savefig("figure/TCGA_perf_compare.pdf", dpi=300, bbox_inches='tight')

plt.show()
# %%

true_tmp = y_test_tensor.numpy()
pred_tmp = mlp_clf.predict(mu_test.detach().cpu())

true_tmp[true_tmp == 13] = 12
pred_tmp[pred_tmp == 13] = 12

labels = list(cancer_2_idx.keys())
labels = np.delete(labels, 13)
labels[12] = 'CRC'

cm = confusion_matrix(y_true=true_tmp, 
                      y_pred=pred_tmp,
                      normalize='true')

# Plotting the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=False, fmt='.2g', cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Classification accuracy'})
plt.plot(range(len(cancer_2_idx)+1), range(len(cancer_2_idx)+1), ls="-", lw=2, color="r")

plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Predicted vs actual cancer type across 32 cancers (N=2,986)')
plt.savefig("figure/TCGA_confusion_VETE.pdf", dpi=300, bbox_inches='tight')
plt.show()

# %%  For proposal demo

import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # Define the first layer with input_size to 16 neurons
        self.layer1 = nn.Linear(input_size, 64)
        # Define the second layer with 16 neurons to 16 neurons
        self.layer2 = nn.Linear(64, 32)
        # Define the third layer with 16 neurons to output_size neurons
        
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, output_size)

    def forward(self, x):
        # Pass the input through the first layer, then apply ReLU activation
        x = F.relu(self.layer1(x))
        # Pass the result through the second layer, then apply ReLU activation
        x = F.relu(self.layer2(x))
        # Pass the result through the third layer
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Number of epochs is the number of times you go through the entire dataset

#%%
# Training loop
num_epochs = 300

mlp_model = MLP(tcga_tensor.shape[1], 33).to(DEVICE)

loss_function = nn.CrossEntropyLoss() 

# Initialize the Adam optimizer with the model parameters and a learning rate
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass: compute the output of the model by passing the inputs through the model
        outputs = mlp_model(inputs.to(DEVICE))
        
        # Compute the loss
        loss = loss_function(outputs, labels.to(DEVICE))
        
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        optimizer.step()
    
    with torch.no_grad():
        (X_test_tensor, y_test_tensor) = next(iter(test_loader))
        outputs = mlp_model(X_test_tensor.to(DEVICE))
        
        # Compute the loss
        loss_testing = loss_function(outputs, y_test_tensor.to(DEVICE))
    
    print(f'Epoch {epoch+1},' + \
        f'Training Loss: {loss.item():.3f},' + \
            f'Testing loss: {loss_testing.item():.3f},' + \
                f'Accuracy: {torch.sum(outputs.argmax(dim=1).cpu() == y_test_tensor)/len(y_test_tensor):.3f}')
print('Training complete')
# %%
from collections.abc import Iterable
class IGNIM:
    def __init__(self, model: nn.Module, n_steps: int = 100, q = 2):
        self.model = model
        self.n_steps = n_steps
        self.device = next(model.parameters()).device
        
        self.q = q

    def attribute(self, x, baselines, target = 0):
        
        baselines = self._baseline_resize(x, baselines)
            
        interpolated = self._interpolate(x, baselines)
        gradients = self._compute_grads(interpolated, target)
        ig = self._integrate(gradients)
        attribution = (x - baselines) * ig
        
        return attribution
    
    def _baseline_resize(self, x, baselines):
        if isinstance(baselines, int) or isinstance(baselines, float):
            baselines = baselines * torch.ones(x.shape, device = self.device)
        elif isinstance(baselines, torch.Tensor) & (len(baselines.shape) == 1) & (len(x.shape) == 2):
            baselines = baselines.repeat(x.shape[0]).to(self.device)
        elif isinstance(baselines, torch.Tensor) & (len(baselines.shape) == len(x.shape)):
            baselines = baselines.to(self.device)
        else:
            raise ValueError
        
        return baselines
    
    def _interpolate(self, x, baselines):
        #x: batch X n_features
        #baseline:  batch X n_features

        alphas = torch.linspace(0.0, 1.0, steps=self.n_steps + 1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        delta = x - baselines
        interpolated = baselines.unsqueeze(0) + alphas * delta.unsqueeze(0)
        return interpolated

    def _compute_grads(self, x, target):
        self.model.eval()  
        clones = x.clone()
        clones.requires_grad_()
        pred = self.model(clones)
        if len(pred.shape) ==3:
            if isinstance(target, Iterable):
                pred = pred[:,list(range(len(target))), target]
            else:
                pred = pred[..., target]
        pred.backward(torch.ones_like(pred, device = self.device))
        return clones.grad.detach()

    def _integrate(self, gradients):
        gradients = torch.abs(gradients) ** self.q
        grads = (gradients[:-1] + gradients[1:]) / 2.
        integral = grads.mean(dim=0)
        return integral ** (1/self.q)

#%%

ignim10 = IGNIM(mlp_model.cpu(), q=10)

# sig10 = IntegratedGradients(ffnn)
# sig_random = IntegratedGradients(ffnn_random)

X_test_tensor[y_test_tensor == 18,:]

attr_OV_vs_BRCA_quantile = ignim10.attribute(X_test_tensor[y_test_tensor == 18,:], 
                                        baselines=X_test_tensor[y_test_tensor == 0,:].mean(dim=0, keepdim = True), 
                                        target = 18).abs()

# %%
attr_OV_vs_BRCA_quantile_mean = attr_OV_vs_BRCA_quantile.mean(dim=0)
# %%

sorted_attr, indices_attr = torch.sort(attr_OV_vs_BRCA_quantile_mean, descending=True)

id2gene_mapping = {}
for key, value in gene2id_mapping.items():
    id2gene_mapping[value] = key

#%%
plt.figure(figsize=(3, 2))
plt.bar(np.array([id2gene_mapping[idx.item()] for idx in indices_attr[:10]]), sorted_attr[:10])
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('Gene')
plt.ylabel('Attribute value')
plt.title("Top 10 genes with highest\nattribute values")
plt.savefig('figure/proposal_TCGA_demo_top_genes.pdf', bbox_inches='tight', dpi=300)
# %%
from scipy.stats import ttest_ind

ttest_ind(X_test_tensor[y_test_tensor == 18, indices_attr[0]],
          X_test_tensor[y_test_tensor == 0 , indices_attr[0]])

ttest_ind(X_test_tensor[y_test_tensor == 18, indices_attr[1]],
          X_test_tensor[y_test_tensor == 0 , indices_attr[1]])
# %%
