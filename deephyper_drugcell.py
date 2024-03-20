from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem

problem = HpProblem()

# Discrete hyperparameter (sampled with uniform prior)
problem.add_hyperparameter((5, 20), "num_epochs", default_value=10)

# Discrete and Real hyperparameters (sampled with log-uniform)
problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
problem.add_hyperparameter((0.1, 10, "log-uniform"), "learning_rate", default_value=5)

print(problem)

#def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
#    method_kwargs = {
#        "num_cpus": 1,
#        "num_cpus_per_task": 1,
#        "callbacks": [TqdmCallback()]
#    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
#    if is_gpu_available:
#        method_kwargs["num_cpus"] = n_gpus
#        method_kwargs["num_gpus"] = n_gpus
#        method_kwargs["num_cpus_per_task"] = 1
#        method_kwargs["num_gpus_per_task"] = 1

#    evaluator = Evaluator.create(
#        run_function,
#        method="ray",
#        method_kwargs=method_kwargs
#    )
#    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
#
#    return evaluator

#evaluator_1 = get_evaluator(quick_run)
