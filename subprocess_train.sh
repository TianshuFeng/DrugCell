#!/bin/bash


drug_tensor=$1
response_data=$2
model_outdir=$3

# activate conda env for model
source activate /homes/ac.rgnanaolivu/miniconda3/envs/drugcell_python

python model_train.py \
  --drug_tensor=$drug_tensor \
  --response_data=$response_data \
  --model_outdir=$model_outdir

