import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from src.utils.map import SinD_map


"""
This script includes classes that read data from different datasets and puts the data 
into a dictionary of the same format.
dict =  {
            _id:    {
                        "x": list[float],
                        "y": list[float],
                        "vx": list[float],
                        "vy": list[float],
                        "ax": list[float],
                        "ay": list[float]
                    }
        }

"""

CWD = os.getcwd()
while CWD.rsplit('/', 1)[-1] != 'Pedestrian_Project':
    CWD = os.path.dirname(CWD)

ROOT = CWD + "/resources"


class SinD:
    """Class that reads the data from SinD dataset.

    Parameters:
    -----------
    name : str
        The name of the file from the SinD dataset that will be read
        and used (default: 'Ped_smoothed_tracks')

    file_extension : str
        The extension of the file (default: '.csv')


    Functions:
    -----------
    data(input_len: int, save_data: bool) -> np.array
        retrieves every input_len part of the trajectories and
        returns a numpy array containing the data inside

    plot_dataset() -> None
        plots both a 2D plot of the historical locations along the
        trajectory, a 3D plot containing the velocity profile, and
        a 3D plot for the acceleration profile
    """

    def __init__(
        self,
        name: str = "Ped_smoothed_tracks",
        file_extension: str = ".csv",
        drop_file: str = None,
    ):
        self._DATADIR = "SinD/Data"
        self._DATASETS = os.listdir("/".join([ROOT, self._DATADIR]))
        if ".DS_Store" in self._DATASETS:
            self._DATASETS.pop(self._DATASETS.index(".DS_Store"))
        self._DATASETS.pop(self._DATASETS.index("mapfile-Tianjin.osm"))
        if drop_file:
            self._DATASETS.pop(self._DATASETS.index(drop_file))
        self.map = SinD_map()
        self.__load_dataset(name + file_extension)

    def __load_dataset(self, name):
        i = 0
        self.frequency = 1 / (100.100100100 / 1000)  # ??
        self.pedestrian_data = {}
        for dataset in self._DATASETS:
            _path = "/".join([ROOT, self._DATADIR, dataset, name])
            _data = pd.read_csv(_path)
            for _id in _data["track_id"].unique():
                ped = _data.loc[_data["track_id"] == _id]
                x, y, vx, vy, ax, ay = (
                    ped["x"],
                    ped["y"],
                    ped["vx"],
                    ped["vy"],
                    ped["ax"],
                    ped["ay"],
                )
                self.pedestrian_data.update(
                    {i: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}
                )
                i += 1
    
    def split_pedestrian_data(self, chunk_size=90, padding_value=1000):
        """
        Split pedestrian data into fixed-size chunks and convert to torch.Tensor,
        padding with -1 where data do not exist.

        :param data: Dictionary of pedestrian data.
        :param chunk_size: Size of each chunk.
        :return: Dictionary with split and padded data, DataFrame with pedestrian IDs.
        """
        split_data = {}
        idx = 0

        for _, attributes in self.pedestrian_data.items():
            # Stack 'x', 'y', 'vx', 'vy', 'ax', 'ay' vertically to get a 2D array for each
            combined_data = np.stack([
                attributes['x'], attributes['y'],
                attributes['vx'], attributes['vy'],
                attributes['ax'], attributes['ay']
            ], axis=1)

            # Calculate the number of chunks and the number of items in the last chunk
            total_length = len(attributes['x'])
            num_chunks = (total_length + chunk_size - 1) // chunk_size  # Ceiling division

            for i in range(num_chunks):
                start_index = i * chunk_size
                end_index = min(start_index + chunk_size, total_length)

                # Slice the combined data for this chunk
                chunk = combined_data[start_index:end_index]

                # Padding if necessary
                if chunk.shape[0] < chunk_size:
                    pad_length = chunk_size - chunk.shape[0]
                    chunk = np.pad(chunk, ((0, pad_length), (0, 0)), mode='constant', constant_values=padding_value)

                # Splitting combined data back into coords, velocity, and acceleration, and converting to tensors
                split_data[idx] = {
                    'coords': torch.tensor(chunk[:, :2], dtype=torch.float32),
                    'velocity': torch.tensor(chunk[:, 2:4], dtype=torch.float32),
                    'acceleration': torch.tensor(chunk[:, 4:], dtype=torch.float32)
                }

                idx += 1

        return split_data

    def plot_dataset(
        self,
        data=None,
        color: str = "orange",
        map_overlay: bool = True,
        alpha: float = 0.2,
    ):
        pedestrian_data = data if data else self.pedestrian_data
        ax1 = plt.figure(1).add_subplot(projection="3d")
        ax2 = (
            self.map.plot_areas(alpha=alpha)[0]
            if map_overlay == True
            else plt.figure(2).add_subplot()
        )
        ax3 = plt.figure(3).add_subplot(projection="3d")
        for _id in pedestrian_data.keys():
            x, y = np.array(pedestrian_data[_id]["x"]), np.array(
                pedestrian_data[_id]["y"]
            )
            vx, vy = pedestrian_data[_id]["vx"], pedestrian_data[_id]["vy"]
            ax, ay = pedestrian_data[_id]["ax"], pedestrian_data[_id]["ay"]
            v = np.sqrt(np.array(vx).T ** 2 + np.array(vy).T ** 2)
            a = np.sqrt(np.array(ax).T ** 2 + np.array(ay).T ** 2)
            ax1.plot(x, y, zs=v, c="r"), ax1.set_title(
                "Velocity profile of trajectories"
            ), ax1.set_xlim(0, 30), ax1.set_ylim(0, 30), ax1.set_zlim(0, 5)
            ax1.set_xlabel("X"), ax1.set_ylabel("Y"), ax1.set_zlabel("V")
            ax2.plot(x, y, c=color), ax2.set_title("Pedestrian trajectories")
            ax3.plot(x, y, zs=a, c="r"), ax3.set_title(
                "Acceleration profile of trajectories"
            ), ax3.set_xlim(0, 30), ax3.set_ylim(0, 30), ax3.set_zlim(0, 5)
            ax3.set_xlabel("X"), ax3.set_ylabel("Y"), ax3.set_zlabel("A")
        plt.grid()
        plt.show()
