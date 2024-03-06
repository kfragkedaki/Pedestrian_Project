#!/usr/bin/env python
import logging
import sys
import os
import time
import torch
import logging

# from src.options import get_options
# from src.utils import load_env
# from src.agents import Agent
from src.options import Options
from src.utils import setup, load_data, register_record, readable_time
from src.model.model import create_model, evaluate, validate, train

ROOT = os.getcwd()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# def get_data(file_path, chunk_size, padding_value):
#     """
#     Load Full Daatset
#     """

#     data_path = os.path.join(ROOT, file_path)

#     if os.path.exists(data_path):
#         logging.info("-------- Loading Dataset --------")

#         sind = SinD()
#         data = sind.split_pedestrian_data(
#             chunk_size=chunk_size, padding_value=padding_value
#         )
#         print(len(data))
#     else:
#         raise FileNotFoundError("No Dataset Available")

#     return data


def run(config):
    total_start_time = time.time()

    # Set the random seed
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run
    
    # Set the device
    use_cuda = torch.cuda.is_available() and not config["no_cuda"]
    use_mps = torch.backends.mps.is_available() and not config["no_cuda"]
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    ## Build and split data
    # data = get_data("resources/sind.pkl", chunk_size=90, padding_value=1000)
    train_loader, val_loader, data = load_data(config, logger)

    # Create model
    model, optimizer, trainer, val_evaluator, start_epoch = create_model(config, train_loader, val_loader, data, logger, device)

    if config["eval_only"]:
        logger.info("Evaluating model ...")
        evaluate(val_evaluator, config, save_embeddings=config['save_embeddings'])
    else:
        logger.info("Starting training...")

        # Train Model
        aggr_metrics_val, best_metrics, best_value = train(model, optimizer, start_epoch,
                                                           trainer, val_evaluator, train_loader, val_loader,
                                                           config)

        # Export record metrics to a file accumulating records from all experiments in the same root file
        register_record(
            config["records_file"],
            config["initial_timestamp"],
            config["experiment_name"],
            best_metrics,
            aggr_metrics_val,
            comment=config["comment"],
        )

        logger.info(
            "Best loss was {}. Other metrics: {}".format(
                best_value, best_metrics
            )
        )
        logger.info("All Done!")
        logger.info(
            "Total runtime: {} hours, {} minutes, {} seconds\n".format(
                *readable_time(time.time() - total_start_time)
            )
        )

        return best_value


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    run(config)
