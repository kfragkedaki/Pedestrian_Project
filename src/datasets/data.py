import os
from multiprocessing import Pool, cpu_count
import re
import logging

import numpy as np
import pandas as pd

from bagpy import bagreader
import shutil

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

    def inverse_normalize(self, df):
        if self.norm_type == "standardization":
            return df * self.std + self.mean
        elif self.norm_type == "minmax":
            return df * (self.max_val - self.min_val) + self.min_val
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return df * grouped.transform("std") + grouped.transform("mean")
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            max_vals = grouped.transform("max")
            return df * (max_vals - min_vals) + min_vals
        else:
            raise NameError(
                f'Inverse normalize method "{self.norm_type}" not implemented'
            )


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


    @staticmethod
    def sort_clean_data(df):
        """"""
        keep_cols = ["track_id", "timestamp_ms", "x", "y", "vx", "vy", "ax", "ay"]

        # sort based on time and id
        df_sorted = df.sort_values(by=["track_id", "timestamp_ms"])

        # make track id unique among different files
        df_sorted["track_id"] = (
            df_sorted["file_id"].astype(str) + "_" + df_sorted["track_id"].astype(str)
        )

        # keep columns
        df_final = df_sorted[keep_cols]

        # remove_stationary_trajectories
        df_final = df_final[
            df_final.groupby("track_id")[["vx", "vy"]].transform(any).all(axis=1)
        ]

        # # remove incorrent datarows
        # df_final = df_final[(df_final["vx"] >= 0) & (df_final["vy"] >= 0)]

        return df_final


    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern
        data_paths = []  # list of all paths
        print('root_dir', root_dir)
        for root, dirs, files in os.walk(root_dir):
            print(root, files)
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

    @staticmethod
    def assign_chunk_idx(df, chunk_len):
        """Assigns a chunk index to each row and trajectory."""
        # Calculate local chunk indices within each unique trajectory
        df["chunk_idx"] = df.groupby("track_id").cumcount() // chunk_len

        # Generate a global chunk ID by enumerating each unique combination of unique_int_id and chunk_idx
        df["data_chunk_len"] = df.groupby(
            ["track_id", "chunk_idx"]
        ).ngroup()  # ngroup assigns unique numbers to each group

        return df

    @staticmethod
    def remove_small_chunks(df, min_size=2):
        """Parameters:
        - df: The dataframe to process.
        - min_size: The minimum number of points a chunk must have to be retained.
        Returns:
        - The filtered dataframe.
        """
        # Group by global_chunk_id and filter
        filtered_df = df.groupby("data_chunk_len").filter(lambda x: len(x) >= min_size)
        return filtered_df


    def reassign_chunk_indices(self, df):
        # Create a unique list of the old chunk indices
        unique_chunks = df['data_chunk_len'].unique()
        # Create a mapping from old to new indices
        chunk_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_chunks)}
        # Map the old indices to new indices
        df['data_chunk_len'] = df['data_chunk_len'].map(chunk_mapping)
        return df



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
        self.feature_names = ["x", "y", "vx", "vy", "ax", "ay"]
        self.all_df = None
        self.all_IDs = None
        self.feature_df = None
        self.max_seq_len = self.config["data_chunk_len"]

    def load_data(self):
        # Load and preprocess data
        self.all_df = self.load_all(
            self.config["data_dir"], pattern=self.config["pattern"]
        )  # 508644

        max_seq_len = self.all_df.groupby(by="track_id").size().max()  # 11726 8_7_1_P1
        self.max_seq_len = (
            self.config["data_chunk_len"] if self.config["data_chunk_len"] != 0 else max_seq_len
        )

        if self.config["data_chunk_len"] is not None:
            self.all_df = self.assign_chunk_idx(self.all_df, self.config["data_chunk_len"])
            # Remove chunks with less than 2 points
            self.all_df = self.remove_small_chunks(self.all_df, min_size=2)
            # Reassign chunk indices
            self.all_df = self.reassign_chunk_indices(self.all_df)
        else:
            self.all_df["data_chunk_len"] = self.all_df["unique_int_id"]

        self.all_df["unique_int_id"], _ = pd.factorize(self.all_df["track_id"])
        self.all_df = self.all_df.set_index("data_chunk_len")
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample (session) IDs # 13088 # CHECK THE TIMESTAMP

        self.feature_df = self.all_df[self.feature_names]

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
        df = SINDData.sort_clean_data(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning(
                "{} nan values in {} will be replaced by 1000".format(num_nan, filepath)
            )
            df = df.fillna(1000)  # NAN VALUES TO 1000

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a set of datasets of various machine sessions."""
        file_name = os.path.basename(os.path.dirname(filepath))
        df = pd.read_csv(filepath)
        df["file_id"] = file_name

        return df

    
class ROSData(BaseData):
    """
    Dataset class for ROS datasets.
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
        self.all_df = self.load_all(
            config["data_dir"], pattern=config["pattern"]
        )  # 508644

        max_seq_len = self.all_df.groupby(by="track_id").size().max()  # 11726 8_7_1_P1
        self.max_seq_len = (
            config["data_chunk_len"] if config["data_chunk_len"] != 0 else max_seq_len
        )

        if config["data_chunk_len"] is not None:
            self.all_df = self.assign_chunk_idx(self.all_df, config["data_chunk_len"])
            # Remove chunks with less than 2 points
            self.all_df = self.remove_small_chunks(self.all_df, min_size=2)
            # Reassign chunk indices
            self.all_df = self.reassign_chunk_indices(self.all_df)
        else:
            self.all_df["data_chunk_len"] = self.all_df["unique_int_id"]

        self.all_df["unique_int_id"], _ = pd.factorize(self.all_df["track_id"])
        self.all_df = self.all_df.set_index("data_chunk_len")
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample (session) IDs # 13088 # CHECK THE TIMESTAMP

        self.feature_names = ["x", "y", "vx", "vy", "ax", "ay"]
        self.feature_df = self.all_df[self.feature_names]


    def load_all(self, root_dir, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe`
        Args:
            root_dir: directory containing all individual .bag files
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # Select paths for training and evaluation
        data_paths = self._gather_data_paths(root_dir, None)

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
                all_df = pd.concat(pool.map(ROSData.load_single, data_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(ROSData.load_single(path) for path in data_paths)

        return all_df


    @staticmethod
    def load_single(filepath):
        df = ROSData.read_data(filepath)
        df = ROSData.sort_clean_data(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning(
                "{} nan values in {} will be replaced by 1000".format(num_nan, filepath)
            )
            df = df.fillna(1000)  # NAN VALUES TO 1000
        return df


    @staticmethod
    def parse_row_to_df(row):
        """Parses a message from a .bag, represented by a pandas row, to a dataframe with one row per id"""

        input_string = row['personstate']
        frame_id = row['header.seq']
        timestamp_ms = row['header.stamp.secs'] * 1000 + row['header.stamp.nsecs'] / 1e6 

        # Remove the outer brackets and split by ', id:' to separate each id block
        input_string = input_string.strip('[]')
        id_blocks = input_string.split(', id:')
        
        # Initialize a list to hold the extracted data
        data = []

        for block in id_blocks:
            # Prepend 'id: ' to the first block
            if not block.startswith('id:'):
                block = 'id: ' + block
            
            # Split the block into lines
            lines = block.split('\n')
            
            # Extract the required fields, ignoring null rows
            if len(lines) > 15: 
                id_val = int(lines[0].split(': ')[1])
                x = float(lines[4].split(': ')[1])
                y = float(lines[5].split(': ')[1])
                vx = float(lines[12].split(': ')[1])
                vy = float(lines[13].split(': ')[1])
                ax = float(lines[14].split(': ')[1])
                ay = float(lines[15].split(': ')[1])
            
                # Append the extracted data as a dictionary to the list
                data.append({
                    'track_id': id_val,
                    'frame_id': frame_id,
                    'timestamp_ms' : timestamp_ms,
                    'x': x,
                    'y': y,
                    'vx': vx,
                    'vy': vy,
                    'ax': ax,
                    'ay': ay
                })

        
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        return df

    
    @staticmethod
    def read_data(filepath):
        """Reads a single .bag."""
        b = bagreader(filepath)
        message_files = b.message_by_topic('/pedestrian_flow_estimate/pedestrian_flow_estimate_reprocessed')

        # Attempt to read the CSV file with UTF-8 encoding
        try:
            df = pd.read_csv(message_files, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, attempt to read with ISO-8859-1 encoding
            df = pd.read_csv(message_files, encoding='ISO-8859-1')

        df = pd.read_csv(message_files)

        if os.path.exists(message_files):
            os.remove(message_files)

        
        # Get the directory path of the file
        directory_path = os.path.dirname(message_files)

        # Check if the directory is empty and remove it
        if os.path.exists(directory_path) and not os.listdir(directory_path):
            shutil.rmtree(directory_path)

        result_df = pd.concat(df.apply(ROSData.parse_row_to_df, axis=1).tolist(), ignore_index=True)

        # Cleaning up timestamps
        result_df['timestamp_ms'] -= min(result_df['timestamp_ms'])

        # Cleaning up ordering
        result_df.sort_values(by=['track_id', 'timestamp_ms'], ascending=[True, True], inplace=True)

        # Cleaning up frame_id
        unique_values = np.unique(result_df['timestamp_ms'])
        sorted_unique = np.sort(unique_values)
        replace_map = {value: rank for rank, value in enumerate(sorted_unique)}
        result_df['frame_id'] = result_df['timestamp_ms'].replace(replace_map).astype(int)

        filename = os.path.basename(filepath)
        result_df["file_id"] = filename

        return result_df

    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern
        print(root_dir)
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
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".bag")
        ]
        if len(input_paths) == 0:
            raise Exception("No .bag files found using pattern: '{}'".format(pattern))
        
        return input_paths



class SVEAData(BaseData):
    """
    Dataset class for online SVEA navigation.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        id_counts: maps trajectory IDs to the number of corresponding columns in all_df
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, config: dict, n_proc=None):

        n_proc = config["n_proc"] if n_proc is None else n_proc
        self.set_num_processes(n_proc=n_proc)
        self.config = config

        # Load and preprocess data
        self.all_df = pd.DataFrame(columns=["x", "y", "vx", "vy", "ax", "ay"])
        self.max_seq_len = config["data_chunk_len"] 
        self.feature_names = ["x", "y", "vx", "vy", "ax", "ay"]
        self.feature_df = self.all_df[self.feature_names]
        self.id_counts = {}


    def process_message(self, msg):
        """
        Process a message and update all_df. Ensure that each trajectory has a max of config['data_chunk_len'] points stored
        """
        frame_id = msg.header.seq
        timestamp_ms = msg.header.stamp.secs * 1000 + msg.header.stamp.nsecs / 1e6 

        for person in msg.personstate:
            track_id = person.id
            x = person.pose.position.x
            y = person.pose.position.y
            vx = person.velocity * np.cos(person.direction)
            vy = person.velocity * np.sin(person.direction)
            ax = 0
            ay = 0

            if track_id not in self.all_IDs:
                self.all_IDs.append(track_id)
                self.id_counts[track_id] = 0

            if self.id_counts[track_id] == self.max_seq_len: 
                just_this_id = self.all_df[self.all_df['track_id'] == track_id]
                self.all_df = self.all_df[~((self.all_df['track_id'] == track_id) & self.all_df[self.all_df['frame_id'] == min(just_this_id['frame_id'])])]
            else:
                self.id_counts[track_id] += 1
            row = [track_id, frame_id, timestamp_ms, x, y, vx, vy, ax, ay]
            self.all_df.append(row)


data_factory = {"sind": SINDData, 'ros' : ROSData, 'svea' : SVEAData}
