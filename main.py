#!/usr/bin/env python
import logging
import sys
import os
import math
import torch
import logging

# from src.options import get_options
# from src.utils import load_env
# from src.agents import Agent
from src.options import Options
from src.utils import setup, load_data
from src.model.model import create_model, evaluate


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
    trainer, val_evaluator, start_epoch = create_model(config, train_loader, val_loader, data, logger, device)
    best_metrics = {}

    if config["eval_only"]:
        logger.info("Evaluating model ...")
        evaluate(val_evaluator, config, save_embeddings=config['save_embeddings'])
    else:
        max_norm = config["max_grad_norm"] if config["max_grad_norm"] > 0 else math.inf
        lr = config["lr"]  # current learning step - when using lr_decay < 1, it changes
        best_value =  1e16  # initialize with +inf due to minimizing of metric (loss)


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    run(config)
