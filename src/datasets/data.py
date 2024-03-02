import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging

import numpy as np
import pandas as pd
import torch
# from datasets.SINDDataset import SinD


logger = logging.getLogger("__main__")


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class SINDData(BaseData):
    """
    Dataset class for SIND dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, config: dict, n_proc=None):

        n_proc = config["n_proc"] if n_proc is None else n_proc
        self.set_num_processes(n_proc=n_proc)
        self.config = config

        # Load and preprocess data
        self.all_df = self.load_all(config["data_dir"], pattern=config["pattern"]) # 508644
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs # 13088 # CHECK THE TIMESTAMP

        self.feature_names = ["x", "y", "vx", "vy", "ax", "ay"]

        max_seq_len = self.all_df.groupby(by='track_id').size().max()
        self.max_seq_len = config['data_chunk_len'] if config['data_chunk_len'] is not None else max_seq_len
        self.tensor_3d = self.create_tensors(chunk_size=config['data_chunk_len'], padding_value=config['padding_value'])

    def load_all(self, root_dir, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe`
        Args:
            root_dir: directory containing all individual .csv files
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # Select paths for training and evaluation
        data_paths = self._gather_data_paths(root_dir, pattern)

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(
                self.n_proc, len(data_paths)
            )  # no more than file_names needed here
            logger.info(
                "Loading {} datasets files using {} parallel processes ...".format(
                    len(data_paths), _n_proc
                )
            )
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(SINDData.load_single, data_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(SINDData.load_single(path) for path in data_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = SINDData.read_data(filepath)
        df = SINDData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning(
                "{} nan values in {} will be replaced by 1000".format(num_nan, filepath)
            )
            df = df.fillna(1000)  # NAN VALUES TO 1000

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions."""
        file_name = os.path.basename(os.path.dirname(filepath))
        df = pd.read_csv(filepath)
        df['file_id'] = file_name

        return df

    @staticmethod
    def select_columns(df):
        """"""
        keep_cols = [
            "track_id",
            "timestamp_ms",
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay"
        ]
        df_sorted = df.sort_values(by=['track_id', 'timestamp_ms'])
        df_sorted['track_id'] = df_sorted['file_id'].astype(str) + '_' + df_sorted['track_id'].astype(str)
        df_final = df_sorted[keep_cols].set_index('timestamp_ms')
        # remove_stationary_data
        df_final = df_final[df_final.groupby('track_id')[['vx', 'vy']].transform(any).all(axis=1)]

        return df_final

    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern

        data_paths = []  # list of all paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                data_paths.append(os.path.join(root, file))

        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        return input_paths

    def create_tensors(self, chunk_size=90, padding_value=1000):
        tensor_list = []
        for _, group in self.all_df.groupby('track_id'):
            # Convert grouped DataFrame to NumPy array, excluding 'global_track_id'
            data = group[self.feature_names].to_numpy()
            # Chunking and padding
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            padded_chunks = [
                np.pad(chunk, ((0, max(0, chunk_size - len(chunk))), (0, 0)), 'constant', constant_values=padding_value) for
                chunk in chunks]
            tensor_list.extend([torch.tensor(chunk, dtype=torch.float32) for chunk in padded_chunks])

        # Stack all tensors to create a single 3D tensor
        return torch.stack(tensor_list)

data_factory = {"sind": SINDData}
