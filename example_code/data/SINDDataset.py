import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from .map import SinD_map


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
while CWD.rsplit("/", 1)[-1] != "Pedestrian_Project":
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
