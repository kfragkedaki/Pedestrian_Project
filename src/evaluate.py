"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys

# 3rd party packages
import torch
from torch.utils.data import DataLoader

# Project modules
from options import Options
from model.model import setup, pipeline_factory, evaluate
from utils import utils
from datasets.data import data_factory, Normalizer
from model.encoder import model_factory
from utils.model_helpers import get_loss_module
from utils.model_helpers import get_optimizer


def main(config):

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"))
    logger.addHandler(file_handler)
    logger.info("Running:\n{}\n".format(" ".join(sys.argv)))  # command used to run

    if config["seed"] is not None:
        torch.manual_seed(config["seed"])

    # Set the device
    use_cuda = torch.cuda.is_available() and not config["no_cuda"]
    use_mps = torch.backends.mps.is_available() and not config["no_cuda"]
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    test_data = data_class(config)
    test_indices = test_data.all_IDs

    # Pre-process features
    if config["normalization"] is not None:
        normalizer = Normalizer(config["normalization"])
        test_data = normalizer.normalize(test_data)

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, test_data)

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(utils.count_parameters(model, trainable=True))
    )

    # Initialize optimizer
    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(model.parameters(), lr=config["lr"])

    # Load model and optimizer state
    if args.load_model:
        model, optimizer, _ = utils.load_model(
            model,
            config["load_model"],
            optimizer,
            config["resume"],
            config["lr"],
            config["lr_step"],
            config["lr_decay"],
        )
    model.to(device)
    loss_module = get_loss_module(config)

    # Initialize data generators
    dataset_class, collate_fn, model_class = pipeline_factory(config)
    test_dataset = dataset_class(test_data, test_indices)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=model.max_len),
    )

    test_evaluator = model_class(
        model,
        test_loader,
        device,
        loss_module,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    evaluate(test_evaluator, config, save_embeddings=True)


if __name__ == "__main__":

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
