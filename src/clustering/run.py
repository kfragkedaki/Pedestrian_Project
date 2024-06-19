import torch
import pandas as pd
import numpy as np

from main import run as run_transformer
from src.utils.config_setup import create_dirs
from src.datasets.data import Normalizer
from src.datasets.plot import SinDMap
from src.clustering.Clusters import HDBSCANCluster
from src.clustering.NearestNeighbor import AnnoyModel
from src.reachability_analysis.labeling_oracle import LabelingOracleSINDData

from src.utils.load_data import load_task_datasets
from src.transformer_model.model import create_model, evaluate
from torch.utils.data import DataLoader

import logging
import logging
import os
import json

ROOT = os.getcwd()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_data(config: dict):
    # Load the data
    pt_file = torch.load(f"{config['output_dir']}/output_data.pt")
    all_data_original, _ = get_original_data(config)

    all_data = np.concatenate(pt_file['targets'], axis=0)
    all_embeddings = np.concatenate(pt_file['embeddings'], axis=0)
    all_embeddings_original = np.concatenate(pt_file['embeddings_original'], axis=0)
    all_predictions = np.concatenate(pt_file['predictions'], axis=0)

    padding_masks = np.concatenate(pt_file['padding_masks'], axis=0)
    target_masks = np.concatenate(pt_file['target_masks'], axis=0)

    return all_data_original, all_data, all_embeddings, all_embeddings_original, all_predictions, padding_masks, target_masks

def get_original_data(config: dict):
    '''
    Load the original data when the data are normalized
    '''
    original_pt_file = torch.load(f"{config['output_dir']}/original_data.pt") # chunk 50, batch size 16, with original data
    chunk_size = config["data_chunk_len"]

    result = []
    feature_names = ["x", "y", "vx", "vy", "ax", "ay"]
    padding_indicators = []

    for key, group in original_pt_file['val_data'].groupby('data_chunk_len'):
        # For each group, convert it into a 2D list (where each sublist is a row in the group)
        chunks = group[feature_names].to_numpy()
        pad_length = chunk_size - chunks.shape[0]
        padded_chunks = np.pad(
            chunks,
            ((0, pad_length), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        # Create an indicator array for padding
        # 0s for original data, 1s for padded data
        indicator_arr = np.zeros((chunks.shape[0], 1) , dtype=int)  # Initialize with zeros, indicating original data

        if pad_length > 0:
            # Create padding indicators (1s) and append to the original indicator array
            padding_indicator = np.ones((pad_length, 1), dtype=int)
            indicator_arr = np.vstack((indicator_arr, padding_indicator))
        
        padding_indicators.append(indicator_arr[:, 0])
        result.append(padded_chunks)

    return np.array(result), np.array(padding_indicators)


def plot_data(padding_masks, all_data_original, all_data, all_predictions, config):
    map = SinDMap()
    batch_id = padding_masks.shape[0] -1

    # orginal data (before normalization, if applied
    print("Original data before normalization (if applicable)")
    map.plot_single_data(pedestrian_data={batch_id: pd.DataFrame(all_data_original[batch_id], columns=["x", "y", "vx", "vy", "ax", "ay"])}, padding_masks=padding_masks)

    # target data (after normalization, if applied)
    print("Target data after normalization (if applicable)")
    if config['data_normalization'] != "none":
        normalizer = Normalizer(norm_type = config['data_normalization'])
        normalized_df = normalizer.normalize(all_data_original)
        map.plot_single_data(pedestrian_data={batch_id: pd.DataFrame(normalized_df[batch_id], columns=["x", "y", "vx", "vy", "ax", "ay"])}, padding_masks=padding_masks)
    map.plot_single_data(pedestrian_data={batch_id: pd.DataFrame(all_data[batch_id], columns=["x", "y", "vx", "vy", "ax", "ay"])}, padding_masks=padding_masks)

    # predicted data before
    print("Predicted data normalized (if applicable) vs unnormalized data")
    if config['data_normalization'] != "none":
        original_df = normalizer.inverse_normalize(all_predictions)
        map.plot_single_data(pedestrian_data={batch_id: pd.DataFrame(original_df[batch_id], columns=["x", "y", "vx", "vy", "ax", "ay"])}, padding_masks=~padding_masks)

    map.plot_single_data(pedestrian_data={batch_id: pd.DataFrame(all_predictions[batch_id], columns=["x", "y", "vx", "vy", "ax", "ay"])})


def load_config(folder='experiments', model_file='SINDDataset_pretrained_2024-04-27_00-11-45_KIP', index=2, index_data=0):

    with open(f'{folder}/{model_file}/configuration.json') as f:
        config = json.load(f)
        config['save_dir'] = ROOT + f'/{folder}/' + config['save_dir'].split('/', index)[-1]
        config['output_dir'] = ROOT + f'/{folder}/' + config['output_dir'].split('/', index)[-1]  + '/eval'
        config['tensorboard_dir'] = ROOT + f'/{folder}/' + config['tensorboard_dir'].split('/', index)[-1] + '/eval'
        config['data_dir'] = ROOT + '/' + config['data_dir'].split('/', index_data)[-1]
        config['load_model'] = config['save_dir'] +'/model_best.pth'
        config['eval_only'] = True
        config['save_embeddings'] = True
        config['val_ratio'] = 1.0
        config['dropout'] = 0.0  # No dropout during evaluation
        config['hyperparameter_tuning'] = False

    create_dirs([config['output_dir']])
    config["original_data"] = False
    config["remove_noise"] = True

    return config


def get_embedding(config: dict, data_oracle: LabelingOracleSINDData):
    # Initialize data generators
    task_dataset_class, collate_fn = load_task_datasets(config)

    # Dataloaders
    val_dataset = task_dataset_class(data_oracle.feature_df, [0])

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=data_oracle.max_seq_len),
    )
    #  Create model, optimizer, and evaluators
    model, optimizer, trainer, val_evaluator, start_epoch = create_model(
        config, None, val_loader, data_oracle, logger, device='cpu'
    )

    # Perform the evaluation
    aggr_metrics, embedding_data = evaluate(val_evaluator, config=config, save_embeddings=True)

    return embedding_data


def get_cluster(config:dict, data_oracle: LabelingOracleSINDData):
    embedding = get_embedding(config, data_oracle)["embeddings"][0]
    nn_model = AnnoyModel(config=config)
    return nn_model.get(embedding)

def run_clusters(config: dict = None, load_embeddings: bool = True, load_clusters: bool = False,
                 min_cluster_size: int = 5, min_samples: int = 30,
                 save_data: bool = True, show_clusters: bool = True, plot_data: bool = False):
    
    if not load_embeddings: run_transformer(config)
    all_data_original, all_data, all_embeddings, all_embeddings_original, all_predictions, padding_masks, target_masks = load_data(config)
    
    if plot_data: plot_data(padding_masks, all_data_original, all_data, all_predictions, config)
    
    cluster_instance = HDBSCANCluster(embeddings=all_embeddings, target=all_data_original,padding_masks=padding_masks, min_cluster_size=min_cluster_size, min_samples=min_samples, config=config)
    
    # Clustering
    if not load_clusters:
        cluster_instance = HDBSCANCluster(embeddings=all_embeddings, target=all_data_original,padding_masks=padding_masks, min_cluster_size=min_cluster_size, min_samples=min_samples, config=config)
        cluster_instance.run(original_data=config["original_data"], remove_noise=config["remove_noise"], save_data=save_data, show_clusters=show_clusters)
        
        # Build and Save Neirest Neighbor Model
        nn_model = AnnoyModel(config=config)
        nn_model.build()
    else:
        cluster_instance = HDBSCANCluster(embeddings=all_embeddings, target=all_data_original,padding_masks=padding_masks, min_cluster_size=min_cluster_size, min_samples=min_samples, config=config)
        data = cluster_instance.load_clusters(original_data=config["original_data"], remove_noise=config["remove_noise"])
                
        return data

    
if __name__ == '__main__':
    # model_file = 'SINDDataset_pretrained_2024-03-26_22-16-45_F2y'
    # index = 6
    # index_data = 5
    # folder = 'ray_results_backpack'
    # Load the configuration

    # model_file = 'SINDDataset_pretrained_2024-04-08_19-13-17_U12'
    # index = 6
    # index_data = 5
    # folder = 'ray_results_original'

    config = load_config(folder='experiments', model_file='SINDDataset_pretrained_2024-04-27_00-11-45_KIP', index=2, index_data=0)
    run_clusters(config=config, load_embeddings=False, load_clusters=False)

    logger.info("Finished Clustering.")