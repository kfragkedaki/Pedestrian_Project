"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import math
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from models.model import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset, save_indices
from models.encoder import model_factory
from models.loss import get_loss_module
from models.optimizers import get_optimizer


def main(config):

    total_epoch_time = 0
    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)
    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        
    # Set the device
    use_cuda = torch.cuda.is_available() and not config['no_cuda']
    use_mps = torch.backends.mps.is_available() and not config['no_cuda']
    device = torch.device('cuda' if use_cuda else "mps" if use_mps else 'cpu')
    
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config, n_proc=config['n_proc'])

    # Split dataset
    train_indices, val_indices = split_dataset(data_indices=my_data.all_IDs, validation_ratio=config['val_ratio'], \
                                                random_seed=config['seed'])
       

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    save_indices(indices={'train': train_indices, 'val': val_indices}, folder=config['output_dir'])
    train_data = my_data.feature_df.loc[train_indices]
    val_data = my_data.feature_df.loc[val_indices]

    # Pre-process features
    if config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        train_data = normalizer.normalize(train_data)
        if len(val_indices):
            val_data = normalizer.normalize(val_data)

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, train_data)

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


    # Initialize optimizer
    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'])

    start_epoch = 0
    max_norm = config['max_grad_norm'] if config['max_grad_norm'] > 0 else math.inf
    lr = config['lr']  # current learning step - when using lr_decay < 1, it changes
    
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_decay'])
    model.to(device)

    loss_module = get_loss_module(config)

    # if config['test_only'] == 'testset':  # Only evaluate and skip training
    #     dataset_class, collate_fn, runner_class = pipeline_factory(config)
    #     test_dataset = dataset_class(test_data, test_indices)

    #     test_loader = DataLoader(dataset=test_dataset,
    #                              batch_size=config['batch_size'],
    #                              shuffle=False,
    #                              num_workers=config['num_workers'],
    #                              pin_memory=True,
    #                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        
    #     if config['extract_embeddings_only']:
    #         embeddings_extractor = runner_class(model, test_loader, device, loss_module,
    #                                         print_interval=config['print_interval'], console=config['console'])
    #         with torch.no_grad():
    #             embeddings = embeddings_extractor.extract_embeddings(keep_all=True)
    #             embeddings_filepath = os.path.join(os.path.join(config["output_dir"] + "/embeddings.pt"))
    #             torch.save(embeddings, embeddings_filepath)
    #         return
    #     else:
    #         test_evaluator = runner_class(model, test_loader, device, loss_module,
    #                                             print_interval=config['print_interval'], console=config['console'])
    #         aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
    #         print_str = 'Test Summary: '
    #         for k, v in aggr_metrics_test.items():
    #             print_str += '{}: {:8f} | '.format(k, v)
    #         logger.info(print_str)
    #         return

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)

    val_dataset = dataset_class(val_data, val_indices)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(train_data, train_indices)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=config['l2_reg'],
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)

    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch, max_norm=max_norm)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)

        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch % config['lr_step'] == 0:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_decay']

            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export record metrics to a file accumulating records from all experiments in the same root file
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
