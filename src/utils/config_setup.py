import json
import os
import sys
import traceback
from datetime import datetime
import string
import random

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    keys_to_delete = ['load_model', 'eval_only', 'pos_encoding', 'pattern', 'data_class', 'data_dir', 
                    'experiment_name', 'comment', 'hyperparameter_tuning', 'output_dir', 'save_embeddings', 
                    'val_ratio', 'dropout', 'exclude_feats']

    for key in keys_to_delete:
        del config[key]

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(load_config(args.config_filepath))
        except:
            logger.critical(
                "Failed to load configuration file. Check JSON syntax and verify that files exist"
            )
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                output_dir
            )
        )

    output_dir = os.path.join(output_dir, config["experiment_name"])

    # Create checkpoint, prediction and tensorboard directories
    config["initial_timestamp"] = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    # if len(config["experiment_name"]) == 0:
    rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    output_dir += "_" + config["initial_timestamp"] + "_" + rand_suffix

    config["output_dir"] = output_dir
    config["save_dir"] = os.path.join(output_dir, "checkpoints")
    config["tensorboard_dir"] = os.path.join(output_dir, "tb_summaries")
    create_dirs([config["save_dir"], config["tensorboard_dir"]])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config
