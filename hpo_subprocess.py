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

# ---------------------
# Model hyperparameters
# ---------------------
problem = HpProblem()

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
# ---------------------
# Some IMPROVE settings
# ---------------------
source = "GDSC"
split = 0
data_tensor = f"ml_data/{source}/gdsc_tensor.csv"
response_data = f"ml_data/{source}/response_gdcs2.csv"
model_outdir = f"out_models_hpo/{source}/"
log_dir = "hpo_logs/"
subprocess_bashscript = "subprocess_train.sh"

@profile
def run(job, optuna_trial=None):

    model_outdir_job_id = model_outdir + f"/{job_id}"
    
    subprocess.run(["bash", subprocess_bashscript,
                    str(data_tensor),
                    str(response_data),
                    str(model_outdir_job_id)], 
                    capture_output=True, text=True, check=True)
    
    f = open(model_outdir + 'val_scores.json')
    val_scores = json.load(f)
    objective = -val_scores["val_loss"]

    # Checkpoint the model weights
    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    # return {"objective": objective, "metadata": metadata}
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)

            search = CBO(
                problem,
                evaluator,
                log_dir=log_dir,
                verbose=1,
            )

            results = search.search(max_evals=100)
