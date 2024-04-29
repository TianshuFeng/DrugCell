"""
Before running hpo.py, first need to preprocess the data.
This can be done by running preprocess_example.sh

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $IMPROVE_DATA_DIR and $PYTHONPATH are set:
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/path/to/IMPROVE_lib
"""
import copy
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
import subprocess
import json
import subprocess
import json
from mpi4py import MPI
import logging
import os
import mpi4py
import os
from datetime import datetime

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print('rank is {0}'.format(rank))
num_gpus_per_node = 4
gpu_id0 = 4
#os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus_per_node + gpu_id0)
os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:" + str(rank % num_gpus_per_node +gpu_id0)
#os.environ["CUDA_VISIBLE_DEVICES"] = 'cuda:5' 


logging.basicConfig(
    filename=f"deephyper.{rank}.log", # optional if we want to store the logs to disk
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)


# ---------------------
# Model hyperparameters
# ---------------------

#problem.add_hyperparameter([5,6],"cuda", default_value=6)
# ---------------------
# Some IMPROVE settings
# ---------------------
source = "GDSC"
#split = 0
current_working_dir = os.getcwd()
#train_ml_data_dir = f"ml_data/{source}/"
#val_ml_data_dir = f"ml_data/{source}/"
train_ml_data_dir = os.path.join(current_working_dir, f"ml_data/{source}/")
val_ml_data_dir = os.path.join(current_working_dir, f"ml_data/{source}/")

model_outdir = os.path.join(current_working_dir, f"out_models_hpo/{source}/")
#log_dir = "hpo_logs/"
log_dir = f"{source}_dh_hpo_logs/"
image_file= os.path.join(current_working_dir + "/images/DrugCell_tianshu:0.0.1-20240422.sif")
subprocess_bashscript = "subprocess_train.sh"
current_date = datetime.now().strftime("%Y-%m-%d")
#train_file = train_ml_data_dir + "/train_data.pt"
#test_file = train_ml_data_dir + "/test_data.pt"

@profile
def run(job, optuna_trial=None):
    print(job)
    job_id = job.id
    epochs = job.parameters['epochs']
    batch_size = job.parameters['batch_size']
    learning_rate = job.parameters['learning_rate']
    direct_gene_weight_param = job.parameters['direct_gene_weight_param']
    num_hiddens_genotype = job.parameters['num_hiddens_genotype']
    num_hiddens_final = job.parameters['num_hiddens_final']
    inter_loss_penalty = job.parameters['inter_loss_penalty']
    eps_adam = job.parameters['eps_adam']
    beta_kl = job.parameters['beta_kl']
    model_outdir_job_id = model_outdir + f"/{job_id}"
    command = f"bash {subprocess_bashscript} {train_ml_data_dir} {val_ml_data_dir} {model_outdir_job_id} {epochs} {batch_size} {learning_rate} {direct_gene_weight_param} {num_hiddens_genotype} {num_hiddens_final} {inter_loss_penalty} {eps_adam} {beta_kl}"
    print(command)
    subprocess_res = subprocess.run(
        [
            "bash", subprocess_bashscript,
            str(train_ml_data_dir),
            str(val_ml_data_dir),
            str(model_outdir_job_id),
            str(epochs),
            str(batch_size),
            str(learning_rate),
            str(direct_gene_weight_param),
            str(num_hiddens_genotype),
            str(num_hiddens_final),
            str(inter_loss_penalty),
            str(eps_adam),
            str(beta_kl)
        ], 
        capture_output=True, text=True, check=True
    )
#    cmd = "singularity exec --nv --bind " +  str(current_working_dir) + " " +  str(image_file) + " " + subprocess_bashscript + " " + str(train_ml_data_dir) + " " + str(val_ml_data_dir) + " " + str(model_outdir_job_id)
#    subprocess_res = subprocess.run(["singularity", "exec", "--nv", "--bind",
#                                     str(current_working_dir),
#                                     str(image_file),
#                                     subprocess_bashscript,
#                                     str(train_ml_data_dir),
#                                     str(val_ml_data_dir),
#                                     str(model_outdir_job_id)], 
#                                    capture_output=True, text=True, check=True)
    print(subprocess_res.stdout)
    print(subprocess_res.stderr)

    f = open(model_outdir_job_id + '/test_scores.json')
    val_scores = json.load(f)
    val_scores = val_scores[0]
    objective = -val_scores["test_loss"]
    print("objective:", objective)

    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    return {"objective": objective, "metadata": val_scores}

if __name__ == "__main__":
    import time
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator
    t0 = time.time()
    problem = HpProblem()
    problem.add_hyperparameter((4, 12), "epochs", default_value=10)
    problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
    problem.add_hyperparameter((0.0001,  1e-2, "log-uniform"), "learning_rate", default_value=0.001)
    problem.add_hyperparameter((0.0, 0.3), "direct_gene_weight_param",default_value=0.1)  # continuous hyperparameter
    problem.add_hyperparameter((0, 10), "num_hiddens_genotype", default_value=6)  # discrete hyperparameter
    problem.add_hyperparameter((0, 10), "num_hiddens_final", default_value=6)  # discrete hyperparameter  
    problem.add_hyperparameter((0.1, 0.5), "inter_loss_penalty", default_value=0.2)
    problem.add_hyperparameter((1e-06, 1e-04), "eps_adam", default_value=1e-05)
    problem.add_hyperparameter((0.0001, 1, "log-uniform"),"beta_kl", default_value=0.001)
    problem.add_hyperparameter(['100,50,6'],"drug_hiddens", default_value="100,50,6")    
    with Evaluator.create(run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}) as evaluator:
        if evaluator is not None:
            print(problem)
            search = CBO(
                problem,
                evaluator,
                log_dir=log_dir,
                verbose=1)
            results = search.search(max_evals=3)
            results = results.sort_values("m:val_scores", ascending=True)
            print(results)            
            results.to_csv(model_outdir + "/hpo_results.csv", index=False)


    print("Finished deephyper HPO.")
