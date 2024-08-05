import pprint as pp

from src.options import Options
from src.utils.config_setup import setup
from main import run as main
from src.utils.hyperparemer_tuning_config import hyperparameter_config

import ray
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session

import os


ROOT = os.getcwd()


def run(hyperparameter_config: dict):
    # Pretty print the run args
    # hyperparameter_config["data_dir"] = ROOT + "/resources/SinD/Data"
    # hyperparameter_config["data_class"] = "sind"
    # hyperparameter_config["pattern"] = "Ped_smoothed_tracks"
    # hyperparameter_config["pos_encoding"] = "learnable"
    # hyperparameter_config["name"] = "SINDDataset_pretrained"
    # hyperparameter_config["comment"] = (
    #     "pretraining_through_imputation-hyperparameter_tuning"
    # )
    # hyperparameter_config["output_dir"] = ROOT + "/ray_results"

    hyperparameter_config["data_dir"] = ROOT + "/bags"
    hyperparameter_config["data_class"] = "sind"
    hyperparameter_config["pattern"] = "aggregated_data"
    hyperparameter_config["pos_encoding"] = "learnable"
    hyperparameter_config["name"] = "SINDDataset_pretrained"
    hyperparameter_config["comment"] = (
        "pretraining_through_imputation-hyperparameter_tuning"
    )
    hyperparameter_config["output_dir"] = ROOT + "/ray_results"

    args_list = [f"--{k}={v}" for k, v in hyperparameter_config.items()]
    args_list.append("--hyperparameter_tuning")
    args_list.append("--harden")

    opts = Options().parse(args_list)

    pp.pprint(vars(opts))

    main(setup(opts), session)


if __name__ == "__main__":
    N_ITER = 1000
    ray.init(num_cpus=12, num_gpus=1)
    searcher = HyperOptSearch(
        space=hyperparameter_config,
        metric="loss",
        mode="min",
        n_initial_points=int(N_ITER / 10),
    )
    algo = ConcurrencyLimiter(searcher, max_concurrent=6)
    objective = tune.with_resources(
        tune.with_parameters(run), resources={"cpu": 12, "gpu": 1}
    )

    tuner = tune.Tuner(
        trainable=objective,
        run_config=air.RunConfig(),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=N_ITER,
        ),
    )

    results = tuner.fit()
    ray.shutdown()
