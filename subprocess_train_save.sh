#!/bin/bash


CONDA_ENV='/homes/ac.tfeng/miniconda3/envs/drugcell_python'
CONDA='/homes/ac.tfeng/miniconda3/condabin/conda'
echo "Allow conda commands in shell script by running 'conda shell.bash hook'"
eval "$(conda shell.bash hook)"
#${CONDA} shell.bash hook
echo "Activated conda commands in shell script"
conda activate $CONDA_ENV
echo "Activated conda env $CONDA_ENV"
export TF_CPP_MIN_LOG_LEVEL=3

train_ml_data_dir=$1
val_ml_data_dir=$2
model_outdir=$3
epochs=$4
batch_size=$5
learning_rate=$6
direct_gene_weight_param=$7
num_hiddens_genotype=$8
num_hiddens_final=$9
inter_loss_penalty=${10}
eps_adam=${11}
beta_kl=${12}
echo "train_ml_data_dir: $train_ml_data_dir"
echo "val_ml_data_dir:   $val_ml_data_dir"
echo "model_outdir:      $model_outdir"


# activate conda env for model
#epochs=20

echo "python DrugCell_train_improve_save.py \\
  --train_ml_data_dir=$train_ml_data_dir \\
  --val_ml_data_dir=$val_ml_data_dir \\
  --model_outdir=$model_outdir \\
  --epochs=$epochs \\
  --batch_size=$batch_size \\
  --learning_rate=$learning_rate \\
  --direct_gene_weight_param=$direct_gene_weight_param \\
  --num_hiddens_genotype=$num_hiddens_genotype \\
  --num_hiddens_final=$num_hiddens_final \\
  --inter_loss_penalty=$inter_loss_penalty \\
  --eps_adam=$eps_adam \\
  --beta_kl=$beta_kl"

python DrugCell_train_improve.py \
  --train_ml_data_dir=$train_ml_data_dir \
  --val_ml_data_dir=$val_ml_data_dir \
  --model_outdir=$model_outdir \
  --epochs=$epochs \
  --batch_size=$batch_size \
  --learning_rate=$learning_rate \
  --direct_gene_weight_param=$direct_gene_weight_param \
  --num_hiddens_genotype=$num_hiddens_genotype \
  --num_hiddens_final=$num_hiddens_final \
  --inter_loss_penalty=$inter_loss_penalty \
  --eps_adam=$eps_adam --beta_kl=$beta_kl \

conda deactivate
echo "Deactivated conda env $CONDA_ENV"
