from src.datasets.data import SINDData, SVEAData, ROSData
from src.utils.poly_process import crosswalk_poly_for_label as cpfl
from src.datasets.plot import SinDMap, SVEAMap
from shapely.geometry import LineString, Point
from math import pi
from tqdm import tqdm
import numpy as np
import pickle
import os 

LABELS = {
    "cross_left": 0,
    "cross_right": 1,
    "cross_straight": 2,
    "cross_illegal": 3,
    "crossing_now": 4,
    "not_cross": 5,
    "unknown": 6,
}

CWD = os.getcwd()
while CWD.rsplit("/", 1)[-1] != "Pedestrian_Project":
    CWD = os.path.dirname(CWD)

ROOT = CWD + "/resources"

class LabelingOracleSINDData(SINDData):
    """
    Extended dataset class for SIND dataset with additional functionalities.
    """
    
    def __init__(self, config: dict, n_proc=None):
        # Initialize the base class with all its setup
        super().__init__(config, n_proc)
        self.map = SinDMap()
        self.config = config

    def create_chunks(self, padding_value=0, save_data: bool = True):
        list = []
        mask_list = []
        for _, group in self.all_df.groupby('track_id'):
            # Convert grouped DataFrame to NumPy array, excluding 'global_track_id'
            data = group[self.feature_names].to_numpy()
            # Chunking and padding
            chunks = [data[i:i + self.config["data_chunk_len"]] for i in range(0, len(data), self.config["data_chunk_len"])]
            
            for chunk in chunks:
                chunk_length = len(chunk)
                padding_length = self.config["data_chunk_len"] - chunk_length
                
                # Pad chunk
                padded_chunk = np.pad(chunk, ((0, padding_length), (0, 0)), 'constant', constant_values=padding_value)
                # Create mask: 1s for original data, 0s for padding
                mask = np.ones(self.config["data_chunk_len"], dtype=int)
                # Set padding area to 0
                mask[chunk_length:] = 0

                # Save data in a list
                mask_list.append(mask == True)
                list.append(padded_chunk)

        # Stack all tensors to create a single 3D tensor
        self.dataset = np.stack(list)
        self.padded_batches = np.stack(mask_list)

        if save_data:
            _f = open(ROOT + "/sind.pkl", "wb")
            pickle.dump(np.array(self.dataset), _f)
            _f = open(ROOT + "/sind_padding.pkl", "wb")
            pickle.dump(np.array(self.padded_batches), _f)

        return self.dataset, self.padded_batches

    def labels(
        self,
        data: np.ndarray,
        save_data: bool = True,
        disable_progress_bar: bool = False,
    ):
        _labels = []
        _crosswalks = cpfl(self.map)
        for _data in tqdm(data, desc="Labeling data", disable=disable_progress_bar):
            _x, _y = _data[:, 0] , _data[:, 1]
            _l = LineString(list(zip(_x, _y)))
            _avg_angle = np.arctan2(sum(_y[2:6] - _y[0:4]), sum(_x[2:6] - _x[0:4]))
            _avg_angle_end = np.arctan2(
                sum(_y[-6:-2] - _y[-4:]), sum(_x[-6:-2] - _x[-4:])
            )
            if (
                (_l.crosses(_crosswalks[0]) or _l.crosses(_crosswalks[2]))
                and not (
                    _l.crosses(self.map.road_poly)
                    or _l.crosses(self.map.intersection_poly)
                    or _l.crosses(self.map.gap_poly)
                )
                and Point((_x[0], _y[0])).within(self.map.sidewalk_poly)
            ):
                if (_avg_angle > pi / 4 and _avg_angle < 3 * pi / 4) or (
                    _avg_angle > -3 * pi / 4 and _avg_angle < -pi / 4
                ):
                    _labels.append(LABELS["cross_straight"])
                elif _avg_angle >= -pi / 4 and _avg_angle <= pi / 4:
                    if _y[0] < 16:
                        _labels.append(LABELS["cross_left"])
                    elif _y[0] >= 16:
                        _labels.append(LABELS["cross_right"])
                else:
                    if _y[0] < 16:
                        _labels.append(LABELS["cross_right"])
                    elif _y[0] >= 16:
                        _labels.append(LABELS["cross_left"])
            elif (
                (_l.crosses(_crosswalks[1]) or _l.crosses(_crosswalks[3]))
                and not (
                    _l.crosses(self.map.road_poly)
                    or _l.crosses(self.map.intersection_poly)
                    or _l.crosses(self.map.gap_poly)
                )
                and Point((_x[0], _y[0])).within(self.map.sidewalk_poly)
            ):
                if (_avg_angle > -pi / 4 and _avg_angle < pi / 4) or (
                    _avg_angle > 3 * pi / 4 or _avg_angle < -3 * pi / 4
                ):
                    _labels.append(LABELS["cross_straight"])
                elif _avg_angle > pi / 4 and _avg_angle < 3 * pi / 4:
                    if _x[0] < 14:
                        _labels.append(LABELS["cross_right"])
                    elif _x[0] >= 14:
                        _labels.append(LABELS["cross_left"])
                else:
                    if _x[0] < 14:
                        _labels.append(LABELS["cross_left"])
                    elif _x[0] >= 14:
                        _labels.append(LABELS["cross_right"])
            elif _l.within(self.map.crosswalk_poly):
                _labels.append(LABELS["crossing_now"])
            elif _l.within(self.map.sidewalk_poly):
                _labels.append(LABELS["not_cross"])
            elif (
                _l.intersects(self.map.road_poly)
                or _l.intersects(self.map.intersection_poly)
                or _l.intersects(self.map.gap_poly)
            ):
                _labels.append(LABELS["cross_illegal"])
            elif Point((_x[0], _y[0])).within(self.map.crosswalk_poly) and not (
                _l.intersects(self.map.road_poly)
                or _l.intersects(self.map.intersection_poly)
                or _l.intersects(self.map.gap_poly)
            ):
                if Point((_x[-1], _y[-1])).within(self.map.sidewalk_poly):
                    _labels.append(LABELS["not_cross"])
                else:
                    _labels.append(LABELS["unknown"])
            elif Point((_x[0], _y[0])).within(self.map.sidewalk_poly) and not (
                _l.intersects(self.map.road_poly)
                or _l.intersects(self.map.intersection_poly)
                or _l.intersects(self.map.gap_poly)
            ):
                if Point((_x[-1], _y[-1])).within(self.map.crosswalk_poly):
                    _angle_diff = angle_between_angles(_avg_angle, _avg_angle_end)
                    if np.abs(_angle_diff) < pi / 4 and (
                        (
                            _avg_angle > pi / 4
                            and _avg_angle < 3 * pi / 4
                            and _avg_angle_end > pi / 4
                            and _avg_angle_end < 3 * pi / 4
                        )
                        or (
                            _avg_angle > -3 * pi / 4
                            and _avg_angle < -pi / 4
                            and _avg_angle_end > -3 * pi / 4
                            and _avg_angle_end < -pi / 4
                        )
                        or (
                            _avg_angle > -pi / 4
                            and _avg_angle < pi / 4
                            and _avg_angle_end > -pi / 4
                            and _avg_angle_end < pi / 4
                        )
                        or (
                            (_avg_angle > 3 * pi / 4 or _avg_angle < -3 * pi / 4)
                            and (
                                _avg_angle_end > 3 * pi / 4
                                or _avg_angle_end < -3 * pi / 4
                            )
                        )
                    ):
                        _labels.append(LABELS["cross_straight"])
                    elif _angle_diff < -pi / 4 and _angle_diff > -3 * pi / 4:
                        _labels.append(LABELS["cross_right"])
                    elif _angle_diff > pi / 4 and _angle_diff < 3 * pi / 4:
                        _labels.append(LABELS["cross_left"])
                    else:
                        _labels.append(LABELS["unknown"])
                else:
                    _labels.append(LABELS["unknown"])
            else:
                _labels.append(LABELS["unknown"])
        if save_data:
            _f = open(ROOT + "/sind_labels.pkl", "wb")
            pickle.dump(np.array(_labels), _f)
        return np.array(_labels)
    

    def filter_paddings(self, dataset: np.ndarray, padded_batches: np.ndarray):
        # Find batches with no padding
        unpadded_batches = np.all(padded_batches, axis=1)  # True only for batches with all 1s (no padding)

        # Filter the dataset to keep only completely unpadded batches
        filtered_data = dataset[unpadded_batches]
    
        return filtered_data
    
def angle_between_angles(a1: float, a2: float):
    """Calculate interior angle between two angles

    Parameters:
    -----------
    a1 : float
        The first heading (angle)
    a2 : float
        The second heading (angle)
    """
    v = np.array([np.cos(a1), np.sin(a1)])
    w = np.array([np.cos(a2), np.sin(a2)])
    return np.math.atan2(np.linalg.det([v, w]), np.dot(v, w))

class LabelingOracleROSData(LabelingOracleSINDData):

    def __init__(self, config: dict, n_proc=None):
        # Initialize the base class with all its setup
        super().__init__(config, n_proc)
        self.map = SVEAMap('seven-eleven.osm')
        self.config = config

    def create_chunks(self, padding_value=0, save_data: bool = True):
        list = []
        mask_list = []
        for _, group in self.all_df.groupby('track_id'):
            # Convert grouped DataFrame to NumPy array, excluding 'global_track_id'
            data = group[self.feature_names].to_numpy()
            # Chunking and padding
            chunks = [data[i:i + self.config["data_chunk_len"]] for i in range(0, len(data), self.config["data_chunk_len"])]
            
            for chunk in chunks:
                chunk_length = len(chunk)
                padding_length = self.config["data_chunk_len"] - chunk_length
                
                # Pad chunk
                padded_chunk = np.pad(chunk, ((0, padding_length), (0, 0)), 'constant', constant_values=padding_value)
                # Create mask: 1s for original data, 0s for padding
                mask = np.ones(self.config["data_chunk_len"], dtype=int)
                # Set padding area to 0
                mask[chunk_length:] = 0

                # Save data in a list
                mask_list.append(mask == True)
                list.append(padded_chunk)

        # Stack all tensors to create a single 3D tensor
        self.dataset = np.stack(list)
        self.padded_batches = np.stack(mask_list)

        # TODO if save data:

        return self.dataset, self.padded_batches
    
    def labels(
        self,
        data: np.ndarray,
        save_data: bool = True,
        disable_progress_bar: bool = False,
    ):
        return np.array([0])


    def filter_paddings(self, dataset: np.ndarray, padded_batches: np.ndarray):
        # Find batches with no padding
        unpadded_batches = np.all(padded_batches, axis=1)  # True only for batches with all 1s (no padding)

        # Filter the dataset to keep only completely unpadded batches
        filtered_data = dataset[unpadded_batches]

        return filtered_data

class LabelingOracleSVEAData(SVEAData):

    def __init__(self, config: dict, n_proc=None):
        # Initialize the base class with all its setup
        super().__init__(config, n_proc)
        self.map = SinDMap()
        self.config = config

    def create_chunks(self, padding_value=0, save_data: bool = True):
        list = []
        mask_list = []
        for _, group in self.all_df.groupby('track_id'):
            # Convert grouped DataFrame to NumPy array, excluding 'global_track_id'
            data = group[self.feature_names].to_numpy()
            # Chunking and padding
            chunks = [data[i:i + self.config["data_chunk_len"]] for i in range(0, len(data), self.config["data_chunk_len"])]
            
            for chunk in chunks:
                chunk_length = len(chunk)
                padding_length = self.config["data_chunk_len"] - chunk_length
                
                # Pad chunk
                padded_chunk = np.pad(chunk, ((0, padding_length), (0, 0)), 'constant', constant_values=padding_value)
                # Create mask: 1s for original data, 0s for padding
                mask = np.ones(self.config["data_chunk_len"], dtype=int)
                # Set padding area to 0
                mask[chunk_length:] = 0

                # Save data in a list
                mask_list.append(mask == True)
                list.append(padded_chunk)

        # Stack all tensors to create a single 3D tensor
        self.dataset = np.stack(list)
        self.padded_batches = np.stack(mask_list)


    def filter_paddings(self, dataset: np.ndarray, padded_batches: np.ndarray):
        # Find batches with no padding
        unpadded_batches = np.all(padded_batches, axis=1)  # True only for batches with all 1s (no padding)

        # Filter the dataset to keep only completely unpadded batches
        filtered_data = dataset[unpadded_batches]

        return filtered_data