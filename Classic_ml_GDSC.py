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
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve

from sklearn.neural_network import MLPClassifier

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
DEVICE = 'cuda:4'

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
bst = xgb.XGBClassifier(n_estimators=50, max_depth=4, 
                        n_jobs=4, objective='binary:logistic',
                        eval_metric = 'auc')
res_bst = bst.fit(X_train, y_top5_train, eval_set = [(X_test, y_top5_test)])

print(np.mean(res_bst.predict(X_train) == y_top5_train))
print(np.mean(res_bst.predict(X_test) == y_top5_test))

roc_auc_score(y_top5_test, res_bst.predict_proba(X_test)[:,1])

#%%
rf = RandomForestClassifier(n_estimators = 10)
rf_res = rf.fit(X_train, y_top5_train)

print(np.mean(rf_res.predict(X_train) == y_top5_train))
print(np.mean(rf_res.predict(X_test) == y_top5_test))

roc_auc_score(y_top5_test, rf_res.predict_proba(X_test)[:,1])

#%%
mlp_clf = MLPClassifier(hidden_layer_sizes=[64], random_state=1, max_iter=300)
mlp_clf.fit(X_train, y_top5_train)

print(np.mean(mlp_clf.predict(X_train) == y_top5_train))
print(np.mean(mlp_clf.predict(X_test) == y_top5_test))

roc_auc_score(y_top5_test, mlp_clf.predict_proba(X_test)[:,1])

#%%

class dcell_vae(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, root, 
                 num_hiddens_genotype, num_hiddens_final, n_class, inter_loss_penalty = 0.2):

        super(dcell_vae, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
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
        
        latent = MultivariateNormal(loc = mu, 
                                    scale_tril=torch.diag_embed(std_dec))
        z = latent.rsample()
        
        recon_mean = self.decoder_affine(z)
        logits = F.softmax(recon_mean, -1)

        return logits, mu, log_var, aux_out_map, term_NN_out_map
    
    def loss_log_vae(self, logits, y, mu, log_var, beta = 0.001):
        # y: true labels
        ori_y_shape = y.shape
        
        class_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), 
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
        
                term_loss = F.cross_entropy(output, 
                                             y, 
                                             reduction = 'none').div(np.log(2)).view(*ori_y_shape)
                inter_loss += term_loss

        return inter_loss

model = torch.load("gdsc_50_top5_"+str(4)+".pt").to(DEVICE)

#%%
recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(torch.tensor(X_test).to(DEVICE))
auc_avg = roc_auc_score(y_top5_test, recon_mean.cpu().detach()[:,1])

#%%

fpr_vete, tpr_vete, thresholds_vete = roc_curve(y_top5_test, recon_mean.cpu().detach()[:,1])
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_top5_test, mlp_clf.predict_proba(X_test)[:,1])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_top5_test, res_bst.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_top5_test, rf_res.predict_proba(X_test)[:,1])


# Plot the ROC curve
plt.figure(figsize=(3.7,3))
plt.plot(fpr_vete, tpr_vete, label='VETE')
plt.plot(fpr_mlp, tpr_mlp, label='MLP', ls='dashed')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost', ls=':')
plt.plot(fpr_rf, tpr_rf, label='Random Forest', ls='-.')
plt.axline( (0,0),slope=1,linestyle='--',color='red')
plt.xlim((-0.02,1.02))
plt.ylim((-0.02,1.02))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Top-5 Binary Classifier\n(N=307)')
plt.legend()
plt.savefig('figure/GDSC_ROC.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%

methods = ["VETE", "MLP", "XGBoost", "Random Forest"]
accu = [0.8993, 0.8658, 0.8729, 0.8025]

plt.figure(figsize = (3,2))
plt.bar(methods, accu)
plt.xlabel('Model Name')
plt.ylabel('AUC-ROC')
plt.ylim([0.7, 0.95])
plt.title('Model Performance Comparison\n(N=307)')
plt.tick_params(axis='x', rotation=45)


plt.savefig("figure/GDSC_top5_perf_compare.pdf", dpi=300, bbox_inches='tight')

plt.show()

#%% TSNE

from sklearn.manifold import TSNE

import seaborn as sns

np.random.seed(0)

tsne = TSNE(n_components=2, perplexity=30)

# Fit the model to the data
tsne_tcga_data = tsne.fit_transform(mu.detach().cpu().numpy())

y_test_tensor_str = [idx_2_cancer[idx.item()] for idx in y_top5_test]

# Unique category labels: 'D', 'F', 'G', ...
color_labels = np.unique(y_test_tensor_str)

# List of RGB triplets
rgb_values = sns.color_palette("husl", len(color_labels))

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))

fig, ax = plt.subplots(1,1, figsize=(3, 2.5))
for category in color_labels:
    mask = np.array(y_test_tensor_str) == category
    ax.scatter(tsne_tcga_data[mask, 0], tsne_tcga_data[mask, 1], s=10,
            color=color_map[category], label=category)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(loc='upper center', bbox_to_anchor=(1.7, 1.07), ncol=3)


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
xgb_predict_drug = res_bst_drug.predict(test_data.numpy())
np.sqrt( np.mean((xgb_predict_drug - test_response.numpy())**2) )
#%%
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

rf_drug = RandomForestRegressor(n_estimators = 10, n_jobs=4, verbose=1)

# rf_res_drug = rf_drug.fit(train_data.numpy(), train_response.numpy())
rf_res_drug = load('rf_res_drug.joblib')

# %%
rf_predict_drug = rf_res_drug.predict(test_data.numpy())

rf_rmse = np.sqrt( np.mean((rf_predict_drug - test_response.numpy())**2) )

# 0.099
# %%
from joblib import dump, load
# dump(rf_res_drug, 'rf_res_drug.joblib') 

# %%
from sklearn.neural_network import MLPRegressor

mlp_reg = MLPRegressor(hidden_layer_sizes=[100], random_state=1, max_iter=300)
# mlp_reg.fit(train_data.numpy(), train_response.numpy())
mlp_reg = load('mlp_res_drug.joblib')
# dump(mlp_reg, 'mlp_res_drug.joblib') 
# %%
mlp_predict_drug = mlp_reg.predict(test_data.numpy())
mlp_rmse = np.sqrt( np.mean((mlp_predict_drug - test_response.numpy())**2) )

#%%
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

drugcell = torch.load("gdsc_drug_epoch_new.pt", map_location=DEVICE)
# gdsc_drug_epoch_fixed
# gdsc_drug_12_epoch
#%%
from scipy.stats import spearmanr

with torch.no_grad():
    recon_mean, mu, log_var, aux_out_map, term_NN_out_map = drugcell(test_data.to(DEVICE)[:8000])
    mse_tmp_testing = F.mse_loss(recon_mean.detach().squeeze().cpu(), test_response[:8000])
    print(torch.sqrt(mse_tmp_testing))
    print(spearmanr(recon_mean.detach().squeeze().cpu().numpy(), test_response.numpy()[:8000]))

# %%  For paper plots
plt.figure(figsize=(3,2))

with torch.no_grad():
    recon_mean, mu, log_var, aux_out_map, term_NN_out_map = drugcell(test_data.to(DEVICE))


sns.boxplot([recon_mean.detach().squeeze().cpu().numpy() - test_response.numpy(),
             mlp_predict_drug - test_response.numpy(), 
             xgb_predict_drug - test_response.numpy(),
             rf_predict_drug - test_response.numpy()])
plt.axhline(y=0, ls='--', color='r')
plt.xticks([0,1,2,3], ["VETE","MLP","XGBoost", "Rnadom Forest"])
plt.xlabel("Model Name")
plt.ylabel("Residual of AUC")
plt.tick_params(axis='x', rotation=45)

# plt.savefig("figure/GDSC_drug_perf_compare.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%train_data.numpy(), train_response.numpy()
with torch.no_grad():
    drugcell.to('cpu')
    logits, mu_train, log_var, aux_out_map, term_NN_out_map = drugcell(train_data.to('cpu'))
    logits, mu_test, log_var, aux_out_map, term_NN_out_map = drugcell(test_data.to('cpu'))

mlp_clf = MLPRegressor(hidden_layer_sizes=[32], random_state=1, max_iter=300)
mlp_clf.fit(mu_train.detach().cpu(), train_response.numpy())
#%%
print(np.mean(np.sqrt( np.mean((mlp_clf.predict(mu_train.detach().cpu()) - train_response.numpy())**2) )))
print(np.mean(np.sqrt( np.mean((mlp_clf.predict(mu_test.detach().cpu()) - test_response.numpy())**2) )))

# mse_tmp_testing = F.mse_loss(recon_mean.detach().squeeze().cpu(), test_response[:8000])

#%%
import scipy 
plt.figure(figsize=(3,2))

drug_cell_residual = recon_mean.detach().squeeze().cpu().numpy() - test_response.numpy()

z_scores = np.abs(scipy.stats.zscore(drug_cell_residual))

# Identify the outliers
outliers = drug_cell_residual[z_scores > 5.7]

# Remove the outliers
drug_cell_residual = drug_cell_residual[~np.isin(drug_cell_residual, outliers)]

sns.violinplot([drug_cell_residual,
             mlp_predict_drug - test_response.numpy(), 
             xgb_predict_drug - test_response.numpy(),
             rf_predict_drug - test_response.numpy()], cut=1)
plt.axhline(y=0, ls='--', color='r')
plt.xticks([0,1,2,3], ["VETE","MLP","XGBoost", "Rnadom Forest"])
plt.xlabel("Model Name")
plt.ylabel("Residual of AUC")
plt.title('Comparison of drug response (AUC)\nprediction performance\n(N=19,271)')
plt.tick_params(axis='x', rotation=45)
plt.savefig("figure/GDSC_drug_perf_compare.pdf", dpi=300, bbox_inches='tight')
plt.show()


#%%
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

print(spearmanr(recon_mean.detach().squeeze().cpu().numpy(), test_response.numpy()))
print(spearmanr(mlp_clf.predict(mu_test.detach().cpu()), test_response.numpy()))
print(spearmanr(mlp_predict_drug , test_response.numpy()))
print(spearmanr(xgb_predict_drug , test_response.numpy()))
print(spearmanr(rf_predict_drug  , test_response.numpy()))

print(pearsonr(recon_mean.detach().squeeze().cpu().numpy(), test_response.numpy()))
print(pearsonr(mlp_clf.predict(mu_test.detach().cpu()), test_response.numpy()))
print(pearsonr(mlp_predict_drug , test_response.numpy()))
print(pearsonr(xgb_predict_drug , test_response.numpy()))
print(pearsonr(rf_predict_drug  , test_response.numpy()))

print(r2_score(y_pred = recon_mean.detach().squeeze().cpu().numpy()[:12500], y_true = test_response.numpy()[:12500]))
print(r2_score(y_pred = mlp_clf.predict(mu_test.detach().cpu()), y_true = test_response.numpy()))
print(r2_score(y_pred = mlp_predict_drug , y_true = test_response.numpy()))
print(r2_score(y_pred = xgb_predict_drug , y_true = test_response.numpy()))
print(r2_score(y_pred = rf_predict_drug  , y_true = test_response.numpy()))

#%%
methods = ["VETE", "MLP", "XGBoost", "Random Forest"]
accu = [r2_score(y_pred = mlp_clf.predict(mu_test.detach().cpu()), y_true = test_response.numpy()), 
        r2_score(y_pred = mlp_predict_drug , y_true = test_response.numpy()),
        r2_score(y_pred = xgb_predict_drug , y_true = test_response.numpy()),
        r2_score(y_pred = rf_predict_drug  , y_true = test_response.numpy())]

err_bar = [0.01, 0.01,0.01,0.01]

plt.figure(figsize = (3,2))
plt.bar(methods, accu)
plt.errorbar(methods, accu, yerr=err_bar, fmt=".", color="C3", capsize = 5.5)

plt.xlabel('Model Name')
plt.ylabel(r'$R^2$')
# plt.ylim([0.7, 0.95])
plt.title('Model Performance Comparison\n(N=19,271)')
plt.tick_params(axis='x', rotation=45)


plt.savefig("figure/GDSC_reg_perf_compare.pdf", dpi=300, bbox_inches='tight')

plt.show()


# %%
plt.figure(figsize=(3,2.6))
plt.scatter(test_response[:8000], mlp_predict_drug[:8000], color = 'k', s=5)
plt.axline((0.4, 0.4), (1,1), c='r', ls = '--')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xlabel('True')
plt.ylabel('Predict')
plt.savefig("figure/GDSC_drugcell_perf_scatter.pdf", dpi=300, bbox_inches='tight')


#%%
import pandas as pd

df = pd.DataFrame({'true': test_response[:8000].squeeze(),
                   'predict': recon_mean[:8000].cpu().detach().squeeze()})

df['true_cut'] = pd.cut(df['true'], np.linspace(0, 1, 11), precision=1)

plt.figure(figsize=(3.25, 2.75))
snsbox_drug = sns.boxplot(data=df, x='true_cut', y='predict')
plt.ylim(0.2,1)
plt.xticks(np.linspace(0, 10, 11)-0.5, labels=np.round(np.linspace(0, 1, 11),1))
plt.xlabel("Actual Drug Response (AUC)")
plt.ylabel("Predicted Drug Response (AUC)")
# snsbox_drug.get_xticks()
plt.title("Drug response prediction under\nactual drug response intervals\n(N=19,271)")

plt.savefig("figure/GDSC_drugcell_perf_box.pdf", dpi=300, bbox_inches='tight')

# %%
from sklearn.manifold import TSNE
import seaborn as sns

np.random.seed(0)

tsne = TSNE(n_components=2, perplexity=30)

# Fit the model to the data
tsne_tcga_data = tsne.fit_transform(mu.cpu().numpy())
#%%

# Unique category labels: 'D', 'F', 'G', ...
# color_labels = np.unique(y_test_tensor_str)

# List of RGB triplets
# rgb_values = sns.color_palette("husl", len(color_labels))
plt.figure(figsize=(3, 2.5))
rgb_values = sns.color_palette("Blues", as_cmap=True)

ax = plt.scatter(tsne_tcga_data[:8000, 0], tsne_tcga_data[:8000, 1], s=20,
        cmap=rgb_values, c=test_response[:8000].squeeze())

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
cbr = plt.colorbar(ax)
cbr.ax.set_ylabel('AUC')
plt.title("t-SNE for latent space of VETE\n(N=19,271)")

plt.savefig('figure/GDSC_drug_tsne.pdf', bbox_inches='tight', dpi=300)
# %%  Individual drug performance
drug_info_data = pd.read_csv('data/GDSC/drug_info.tsv', sep='\t')

gdsc_data_train_index = response_gdcs2[torch.isin(response_gdcs2[:,0], train_gdcs_idx)].float()

gdsc_data_test_index = response_gdcs2[torch.isin(response_gdcs2[:,0], test_gdcs_idx)].float()
drug_key_list = [chem_row_id_key[idx.item()] for idx in torch.unique(gdsc_data_test_index[:,1])]

#%%
drugcell.to('cpu')
with torch.no_grad():
    recon_mean, _, _, _, _ = drugcell(train_data.to('cpu'))
#%%
print(r2_score(y_pred = recon_mean.detach().squeeze().cpu().numpy()[8000:], 
               y_true = train_response.numpy()[8000:]))

#%%

pred_drugcell = recon_mean.detach().cpu().squeeze()
residual_drugcell = pred_drugcell.numpy() - train_response.numpy()
drug_list = torch.unique(gdsc_data_train_index[:,1])

indi_drug_rmse = []
indi_drug_pearson = []
indi_drug_pearson_p_value = []
for drug_item in drug_list:
    with torch.no_grad():
        indi_drug_rmse.append(np.sqrt(np.mean((residual_drugcell[gdsc_data_train_index[:,1] == drug_item ])**2)))
        if torch.sum(gdsc_data_train_index[:,1] == drug_item) > 1:
            pearson_test = pearsonr(pred_drugcell[gdsc_data_train_index[:,1] == drug_item ],
                                    train_response.numpy()[gdsc_data_train_index[:,1] == drug_item ])
            indi_drug_pearson.append(pearson_test.statistic)
            indi_drug_pearson_p_value.append(pearson_test.pvalue)
        else:
            indi_drug_pearson.append(0)

indi_drug_rmse = np.array(indi_drug_rmse)
indi_drug_pearson = np.array(indi_drug_pearson)
indi_drug_pearson_p_value = np.array(indi_drug_pearson_p_value)


#%%
sns.lineplot(x = np.array(range(len(indi_drug_rmse)))[:-3], y = np.sort(np.array(indi_drug_rmse))[:-3])

# %%

pearson_order = np.argsort(-np.array(indi_drug_pearson))
a = sns.lineplot(x = np.array(range(len(indi_drug_rmse)))[:-1], y = indi_drug_pearson[pearson_order][:-1])
# sns.scatterplot(x = np.array(range(len(indi_drug_rmse))), 
#                 y = indi_drug_pearson[pearson_order],
#                 hue = indi_drug_pearson_p_value >= 0.05)
c = plt.axhline(y=0, color='k')

line = a.get_lines()

# plt.fill_between(a.get_xdata(), a.get_ydata(), c.get_ydata(), color='blue', alpha=.5)
plt.fill_between(line[0].get_xdata(), line[0].get_ydata(), 0, color='r', alpha=.5)

# plt.yticks([0, 0.4, 0.8])
# plt.ylim(-0.1,0.9)
# %% Top performance drugs
name_chem_id_match = drug_info_data[drug_info_data.improve_chem_id.isin(drug_key_list)][['NAME', 'improve_chem_id']].drop_duplicates(['improve_chem_id'])

drug_list = np.array(drug_list)
top_drug_id = [chem_row_id_key[key] for key in drug_list[pearson_order][:10]]
top_drug_name = [name_chem_id_match[name_chem_id_match['improve_chem_id'] == id]['NAME'].item() for id in top_drug_id]


# %%
plt.figure(figsize=(3, 2))
plt.bar(np.array(top_drug_name), indi_drug_pearson[pearson_order][:10])
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('Drugs')
plt.ylabel('Correlation coefficient\n (predicted vs actual)')
plt.title("Top 10 drugs with highest\nprediction accuracy")
plt.savefig('figure/GDSC_drug_individual_pearson.pdf', bbox_inches='tight', dpi=300)
# %%
drug_list = np.array(drug_list)
bottom_drug_id = [chem_row_id_key[key] for key in drug_list[pearson_order][-10:]]
bottom_drug_name = [name_chem_id_match[name_chem_id_match['improve_chem_id'] == id]['NAME'].item() for id in bottom_drug_id]


# %%
plt.figure(figsize=(3, 2))
plt.bar(np.array(bottom_drug_name), indi_drug_pearson[pearson_order][-10:])
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('Drugs')
plt.ylabel('Correlation coefficient\n (predicted vs actual)')
plt.title("Top 10 drugs with worst prediction accuracy")
plt.savefig('figure/GDSC_drug_individual_pearson_low.pdf', bbox_inches='tight', dpi=300)
# %%
