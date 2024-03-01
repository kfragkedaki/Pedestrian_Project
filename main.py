#!/usr/bin/env python
from datasets.SINDDataset import SinD
import torch
import logging
import warnings
import os

from src.options import get_options
from src.utils import load_env
from src.agents import Agent

ROOT = os.getcwd()

logging.basicConfig(
    format="%(levelname)s: %(message)s", encoding="utf-8", level=logging.DEBUG
)
warnings.filterwarnings("ignore")


def get_data(file_path, chunk_size, padding_value):
    """
    Load Full Daatset
    """

    data_path = os.path.join(ROOT, file_path)

    if os.path.exists(data_path):
        logging.info("-------- Loading Dataset --------")

        sind = SinD()
        data = sind.split_pedestrian_data(
            chunk_size=chunk_size, padding_value=padding_value
        )
        print(len(data))
    else:
        raise FileNotFoundError("No Dataset Available")

    return data


def run(opts: dict()):
    # Set the random seed
    torch.manual_seed(opts.seed)

    # Initialize the Environment
    env = load_env()

    # Train the Agent
    agent = Agent(opts, env)
    agent.train()


if __name__ == "__main__":
    # NOTE: Get the full dataset for best results

    data = get_data("resources/sind.pkl", chunk_size=90, padding_value=1000)

    run(get_options())
