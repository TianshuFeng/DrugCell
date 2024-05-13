#!/bin/bash


CONDA_ENV='/homes/ac.tfeng/miniconda3/envs/drugcell_python'
CONDA='/homes/ac.tfeng/miniconda3/condabin/conda'

echo "Allow conda commands in shell script by running 'conda shell.bash hook'"
eval "$(conda shell.bash hook)"
#${CONDA} shell.bash hook
echo "Activated conda commands in shell script"
conda activate $CONDA_ENV
echo "Activated conda env $CONDA_ENV"
which conda