import pandas as pd
import numpy as np
import os
import pickle
from annoy import AnnoyIndex

ROOT = os.getcwd()
ROOT_RESOURCES = os.path.join(ROOT, 'resources')


class ApproximateNearestNeighbors(object):
    """Abstract base class for neirest neighbor models."""
    def __init__(self, config=None):
        self.config = config
        self.original_data = config["original_data"]


        if config is not None and os.path.exists(config['output_dir'] + '/clusters'):
            self.file_root = os.path.join(config['output_dir'], 'clusters')
        else:
            self.file_root = ROOT_RESOURCES

        self.load_data()


    def build_index(self, renove_noise):
        pass

    def save_index(self, filename):
        pass

    def load_index(self, filename):
        pass

    def query(self, data, k):
        pass

    def load(self, filename):
        filepath = os.path.join(self.file_root, filename)
        
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
       
        return data

    def load_data(self):
        print("Load data from:", self.file_root)
        data = self.load(f'data_{"original"if self.original_data else "embeddings"}.pkl')
        self.data = np.mean(data, axis=1) # pool mean of the embeddings along the time dimension
        self.dimension = self.data.shape[1]
        self.clusters = self.load(f'cluster_labels{"_original" if self.original_data else ""}.pkl')
        
        self.data = self.data[self.clusters != -1]
        self.clusters = self.clusters[self.clusters != -1]
                

class AnnoyModel(ApproximateNearestNeighbors):
    def __init__(self, config=None, n_trees=10, metric='euclidean'):
        super().__init__(config)
        self.n_trees = n_trees
        self.metric = metric
        self.index = None

    def build_index(self):
        self.index = AnnoyIndex(self.dimension, self.metric)
        for i, vector in enumerate(self.data):
            self.index.add_item(i, vector)
        self.index.build(self.n_trees)

    def save_index(self, filename: str = 'annoy_index.ann'):
        filepath = os.path.join(self.file_root, "original_" if self.config["original_data"] else "" + filename)

        self.index.save(filepath)

    def load_index(self, filename: str = 'annoy_index.ann'):
        filepath = os.path.join(self.file_root, filename)
        print("Load Annoy Model from:", filepath)

        self.index = AnnoyIndex(self.dimension, self.metric)
        self.index.load(filepath)

    def query(self, data, k=1):
        return self.index.get_nns_by_vector(data, k, include_distances=True)
    
    def build(self):
        self.build_index()
        self.save_index()

    def get(self, data: np.array):
        self.load_index()
        if data.ndim == 3 and data.shape[0] == 1: data = data[0]
        elif data.ndim == 3 and data.shape[0] != 1: print("Data shape must be (1, n, m) or (n, m)")
        
        if data.ndim == 2: data = data.mean(0)
        indexes, distances = self.query(data)
        predicted_cluster = self.clusters[indexes[0]]

        return predicted_cluster
    