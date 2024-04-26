import pypolycontain as pp
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from typing import List, Union
from src.reachability_analysis.labeling_oracle import LABELS
from itertools import chain



def separate_data_to_class(data: np.ndarray, classification: np.ndarray, size: int) -> np.ndarray:
    """ Separate the entire dataset into a list[list] where each nested list contain 
        the trajectories for that specific class.

        Parameters:
        -----------
        data : np.ndarray
            The entire dataset
        classification : np.ndarray
            The labels from the classification framework for the dataset in data
    """
    # _class = [0] * size
    _class = {}
    for i in range(size):
        _class[i] = []
    # for _i in range(len(_class)): _class[_i] = []
    for i,_trajectory in enumerate(data):
        _class[classification[i]].append(_trajectory)
        
    for i in _class.keys():
        _class[i] = np.array(_class[i])
    return _class


def structure_input_data(data: np.ndarray, labels: np.ndarray):
    """Drops random trajectories such that the data for each class
    is of the same length

    Parameters:
    -----------
    data : np.ndarray
        The chunks from the dataset
    labels: np.ndarray
        The true labels for the data
    """
    _d = {}.fromkeys(labels)
    [_d.update({i: []}) for i in _d.keys()]
    [_d[_l].append(data[i]) for i, _l in enumerate(labels)]
    _lens = [len(v) for v in _d.values()]
    _min_len = min(_lens)
    new_d = []
    new_l = []
    for _l, _v in _d.items():
        _v = np.array(_v)
        _ids = np.random.randint(0, len(_v), size=_min_len)
        new_d = [*new_d, *_v[_ids]]
        new_l = [*new_l, *[_l] * _min_len]
    return np.array(new_d), np.array(new_l)


def create_io_state(data: dict, measurement: pp.zonotope, vel: np.ndarray, classification: Union[int, List[int]], drop_equal: bool = True, angle_filter: bool = True, clustering: bool = False) -> List[np.ndarray]:
    """ Function to create D = (X-, X+, U-) in the reachability algorithm

    Parameters:
    -----------
    data : dict
        Data from that has been precomputed by the separate_data_to_class function, 
        where each key is a class and each value is a 3D array (trajectories, time, features)
    measurement : pp.zonotope
        The measurement from which the reachable sets should be calculated
    vel : np.ndarray
        Current velocity vector of measurement
    classification : int | List[int]
        The classification for the current pedestrian as an int corresponding
        to the class OR the list of all possible classes, in which case the
        function returns all trajectories near the pedestrian regardless of
        class
    input_len : int (default = 30)
        The input length for the chunks of each trajectory
    drop_equal : bool (default = True)
        Determines if it drops data that is equal
    angle_filter : bool (default = True)
        Filter based on initial heading of chunk and pedestrian
    """
    if isinstance(classification, list):
        _data = np.concatenate([data[cls] for cls in classification if cls in data], axis=0)
    else:
        _data = data.get(classification, np.array([]))

    X_m, X_p, U = np.array([]), np.array([]), np.array([])
    # _ped_poly = Polygon(pp.to_V(measurement))
    # _angle_set = np.array([np.arctan2(*vel)-np.pi/8, np.arctan2(*vel)+np.pi/8]) if angle_filter else np.array([np.arctan2(*vel)-np.pi, np.arctan2(*vel)+np.pi])

    for _t in _data:
        _x, _y = _t[:, 0], _t[:, 1]
        _vx, _vy = _t[:, 2], _t[:, 3]
        # _v_avg = np.array([sum(_vx[0:3]), sum(_vy[0:3])])
        # _line = LineString(list(zip(_x, _y)))
        # if (_line.intersects(_ped_poly) and __in_between(np.arctan2(_v_avg[1], _v_avg[0]), _angle_set)) or clustering:
        _X = np.array([_x, _y])
        _X_p, _X_m = _X[:, 1:], _X[:, :-1]
        _U = np.array([_vx, _vy])[:, :-1]
        X_p = np.hstack([X_p, _X_p]) if X_p.size else _X_p
        X_m = np.hstack([X_m, _X_m]) if X_m.size else _X_m
        U = np.hstack([U, _U]) if U.size else _U

    if drop_equal:
        X_p, _ids = __drop_equal(X_p)
        X_m = np.delete(X_m, _ids, axis=1)
        U_d = np.delete(U, _ids, axis=1)

    return [U_d, X_p, X_m, U]


def __in_between(val: float, range: np.ndarray):
    assert range.shape[0] == 2
    return (val > range[0] and val < range[1])

def __drop_equal(arr: np.ndarray):
    _d, _ids = {}, np.array([], dtype=int)
    if len(arr.shape) == 1: return arr, _ids
    assert arr.shape[1] > arr.shape[0]
    for i, a in enumerate(arr.T):
        if str(a) not in _d:
            _d.update({str(a):0})
        elif str(a) in _d:
            _ids = np.hstack((_ids, i))
    return np.delete(arr, _ids, axis=1), _ids

def split_io_to_trajs(X_p: np.ndarray, X_m: np.ndarray, U: np.ndarray, threshold: float = 0.8, dropped: bool = True, N: int = 30):
    """ Split the IO state (that drops equal points) into trajectories of different sizes

        Parameters:
        -----------
        X_p : np.ndarray
            X+ data
        X_m : np.ndarray
            X- data
        U : np.ndarray
            Inputs
        threshold : float (default = 0.8)
            Threshold for when regarding two points on the same trajectory
        dropped : bool (default = True)
            Set this to True if the equal points have been dropped from
            the data
        N : int (default = 30)
            Time horizon of the reachability analysis
    """
    _X_p, _X_m, _U = [], [], []
    if dropped:
        x_prev = X_p[:,0]
        i_prev = 0
        for i,x in enumerate(X_p[:,1:].T):
            _dist = np.linalg.norm(x-x_prev)
            x_prev = x
            if _dist > threshold:
                _X_p.append(X_p[:,i_prev:i+1])
                _X_m.append(X_m[:,i_prev:i+1])
                _U.append(U[:,i_prev:i+1])
                i_prev = i+1
    else:
        for i in range(N, U.shape[1]+1, N):
            _U.append(U[:,i-N:i])
    if len(_U) == 0:
        _X_p.append(X_p[:,i_prev:i+1])
        _X_m.append(X_m[:,i_prev:i+1])
        _U.append(U[:,i_prev:i+1])
    return _X_p, _X_m, _U
    