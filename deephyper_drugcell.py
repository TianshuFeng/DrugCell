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
import time
import copy
from mpi4py import MPI
import tqdm
from tqdm import tqdm as tqdm_bar 
from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np
import sklearn
import os
import torch.optim as optim
from torchmetrics.functional import mean_absolute_error
from scipy.stats import spearmanr
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
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
     force=True,
)

def train(job, optimizer, model, term_mask_map, train_loader):
     model.train()
     DEVICE='cuda:' + str(job.parameters['cuda'])
     print(DEVICE)
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
               recon_mean=recon_mean, y=response.to(DEVICE), mu=mu, log_var=log_var, beta=job.parameters['beta_kl']
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


def get_run(job, params, num_drugs, gdsc_data_train, gdsc_data_test):
     betas_adam=(0.9, 0.99)
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     train_loader = DataLoader(gdsc_data_train, batch_size=job.parameters['batch_size'], shuffle=True)
     test_loader = DataLoader(gdsc_data_test, batch_size=job.parameters['batch_size'], shuffle=False)
     dG, root, term_size_map,term_direct_gene_map, num_genes  = process_drugcell_inputs(params)
     num_hiddens_drug = list(map(int, job.parameters['drug_hiddens'].split(',')))
     model = Drugcell_Vae(term_size_map, term_direct_gene_map, dG, num_genes, num_drugs, 
                          root, job.parameters['num_hiddens_genotype'], num_hiddens_drug, job.parameters['num_hiddens_final'], 
                          inter_loss_penalty=job.parameters['inter_loss_penalty'],
                          n_class = 0)
     
     DEVICE='cuda:' + str(job.parameters['cuda'])
     model.to(DEVICE)
     print(job.parameters['learning_rate'])
     print(betas_adam)
     print(job.parameters['eps_adam'])
     optimizer = torch.optim.Adam(model.parameters(), lr=job.parameters['learning_rate'],
                                  betas=betas_adam,
                                  eps=job.parameters['eps_adam'])
     optimizer.zero_grad()
     term_mask_map = create_term_mask(model.term_direct_gene_map, num_genes, device = DEVICE)
     for name, param in model.named_parameters():
          term_name = name.split('_')[0]
          
          if '_direct_gene_layer.weight' in name:
               param.data = torch.mul(param.data, term_mask_map[term_name].to(DEVICE)) * job.parameters['direct_gene_weight_param']
          else:
               param.data = param.data * job.parameters['direct_gene_weight_param']
     num_epochs = int(job.parameters["num_epochs"])

     for _ in tqdm_bar(range(1, num_epochs + 1), total=num_epochs):
          train(job, optimizer, model, term_mask_map, train_loader)
#     for _ in range(1, int(job.parameters["num_epochs"]) + 1):
#          train(job, optimizer, model, term_mask_map, train_loader)        

     test_loss = evaluate(model, test_loader)
     print(test_loss)
     return test_loss
#    return run

def run(job):
    print(f"Running job: {job.id} with params = {job.parameters}", flush=True)
    fdir = Path('__file__').resolve().parent
    data_dir = str(fdir) + '/hpo_data/'
    params = initialize_parameters()
    params = load_params(params, data_dir)
    num_drugs, gdsc_data_train, gdsc_data_test = preprocess_data(params)
    training_duration = np.random.uniform(low=0, high=1)
    time.sleep(training_duration)
    get_run(job, params, num_drugs, gdsc_data_train, gdsc_data_test)
    return job.parameters["num_epochs"] + job.parameters["batch_size"] + job.parameters["learning_rate"]



def main():
  problem = HpProblem()
  log_dir = "hpo_logs/"
  problem.add_hyperparameter((4, 12), "num_epochs", default_value=10)
  problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
  problem.add_hyperparameter((0.0001, 1, "log-uniform"), "learning_rate", default_value=0.001)
  problem.add_hyperparameter((0.0, 0.3), "direct_gene_weight_param",default_value=0.1)  # continuous hyperparameter
  problem.add_hyperparameter((0, 10), "num_hiddens_genotype", default_value=6)  # discrete hyperparameter
  problem.add_hyperparameter((0, 10), "num_hiddens_final", default_value=6)  # discrete hyperparameter  
  problem.add_hyperparameter((0.1, 0.5), "inter_loss_penalty", default_value=0.2)
  problem.add_hyperparameter((1e-06, 1e-04), "eps_adam", default_value=1e-05)
  problem.add_hyperparameter((0.0001, 1, "log-uniform"),"beta_kl", default_value=0.001)
  problem.add_hyperparameter(['100,50,6'],"drug_hiddens", default_value="100,50,6")
  problem.add_hyperparameter([5,6],"cuda", default_value=6)
  
  with Evaluator.create(run, method="process",
                        method_kwargs={"callbacks": [TqdmCallback()], "num_workers": 2},) as evaluator:
       if evaluator is not None:
            print(problem)
            search = CBO(problem, evaluator, random_state=42, log_dir=log_dir, verbose=1)
            results = search.search(max_evals=10)
            print(results)  


if __name__ == "__main__":
  main()
