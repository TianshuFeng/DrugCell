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
from mpi4py import MPI
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
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.nas.run import run_base_trainer
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from drugcell_GDSC_script_HPO import create_term_mask
from drugcell_GDSC_script_HPO import run_train_vae
from drugcell_GDSC_script_HPO import load_params
from drugcell_GDSC_script_HPO import process_drugcell_inputs
from drugcell_GDSC_script_HPO import initialize_parameters
from drugcell_GDSC_script_HPO import preprocess_data
from drugcell_GDSC_script_HPO import Drugcell_Vae
import ConfigSpace.hyperparameters as csh
from pathlib import Path


def train(params, model, train_loader):
     model.train()
     DEVICE='cuda:' + str(params['cuda'])
     train_predict = torch.zeros(0, 0).to(DEVICE)
     tloader = tqdm.tqdm(enumerate(train_loader))
     for i, (data, response) in tloader:
       optimizer.zero_grad()  # zero the gradient buffer
       recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(data.to(DEVICE))
       if train_predict.size()[0] == 0:
         train_predict = aux_out_map["final"].data
       else:
         train_predict = torch.cat([train_predict, aux_out_map["final"].data], dim=0)
         
         total_loss = 0
         loss_vae = model.loss_log_vae(
              recon_mean=recon_mean, y=response.to(DEVICE), mu=mu, log_var=log_var, beta=params['beta_kl']
         )

         loss_intermidiate = model.intermediate_loss(aux_out_map, response.to(DEVICE))
         total_loss = torch.mean(loss_vae + model.inter_loss_penalty * loss_intermidiate)
         tmp_loss = total_loss.item()
         total_loss.backward()
         for name, param in model.named_parameters():
              if "_direct_gene_layer.weight" not in name:
                   continue
              term_name = name.split("_")[0]
              param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])        
              optimizer.step()

def evaluate(model, test_loader):
     model.eval()
     test_loss = 0
     with torch.no_grad():
          tloader = tqdm.tqdm(enumerate(test_loader))
          for i, (data, response) in tloader:
               recon_mean, mu, log_var, aux_out_map, term_NN_out_map = model(data.to(DEVICE))
               mse_tmp_testing = F.mse_loss(recon_mean.detach().squeeze().cpu(), response.squeeze())
               test_loss = mse_tmp_testing.item()
     return test_loss


def get_run(params, num_drugs, gdsc_data_train, gdsc_data_test):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     train_loader = DataLoader(gdsc_data_train, batch_size=params['batch_size'], shuffle=True)
     test_loader = DataLoader(gdsc_data_test, batch_size=params['batch_size'], shuffle=False)
     dG, root, term_size_map,term_direct_gene_map, num_genes  = process_drugcell_inputs(params)
     num_hiddens_drug = list(map(int, params['drug_hiddens'].split(',')))
     model = Drugcell_Vae(term_size_map, term_direct_gene_map, dG, num_genes, num_drugs, 
                          root, params['num_hiddens_genotype'], num_hiddens_drug, params['num_hiddens_final'], 
                          inter_loss_penalty=params['inter_loss_penalty'],
                          n_class = 0)
     
     DEVICE='cuda:' + str(params['cuda'])
     model.to(DEVICE)
     term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes, device = DEVICE)
     for name, param in model.named_parameters():
          term_name = name.split('_')[0]
          
          if '_direct_gene_layer.weight' in name:
               param.data = torch.mul(param.data, term_mask_map[term_name].to(DEVICE)) * params['direct_gene_weight_param']
          else:
               param.data = param.data * params['direct_gene_weight_param']
#        train(model, criterion, optimizer, train_dataloader)
#        
#        accu_test = evaluate(model, valid_dataloader)
#      return accu_test
#    return run

def run(job):
    print(f"Running job: {job.id} with params = {job.parameters}", flush=True)
    time.sleep(1)
    return job.parameters["x_float"] + job.parameters["x_int"] + job.parameters["x_cat"]



def main():
  fdir = Path('__file__').resolve().parent
  data_dir = str(fdir) + '/hpo_data/'
  params = initialize_parameters()
  params = load_params(params, data_dir)
  num_drugs, gdsc_data_train, gdsc_data_test = preprocess_data(params)
  problem = HpProblem()
  problem.add_hyperparameter((5, 20), "num_epochs", default_value=10)
  problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
  problem.add_hyperparameter((0.1, 10, "log-uniform"), "learning_rate", default_value=5)
  problem.add_hyperparameter((0.0, 10.0), "x_float")  # continuous hyperparameter
  problem.add_hyperparameter((0, 10), "x_int")  # discrete hyperparameter
  problem.add_hyperparameter([0, 2, 4, 6, 8, 10], "x_cat")  # categorical hyperparameter

#  get_run(params, num_drugs, gdsc_data_train, gdsc_data_test)
  with Evaluator.create(run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}) as evaluator:
       if evaluator:
            if evaluator is not None:
                 evaluator = Evaluator.create(
                      run,
                      method="process",
                      method_kwargs={
                           "num_workers": 2,
                      },    )

    # define your search and execute it
  search = CBO(problem, evaluator, random_state=42)

  results = search.search(max_evals=100)
  print(results)  


if __name__ == "__main__":
  main()
