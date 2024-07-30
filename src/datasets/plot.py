import os
import osmium as osm
import math
import pyproj
import matplotlib.pyplot as plt
import re
import numpy as np
import alphashape
from descartes import PolygonPatch
from scipy.ndimage import rotate, shift
from skimage.transform import resize

CWD = os.getcwd()
while CWD.rsplit("/", 1)[-1] != "Pedestrian_Project":
    CWD = os.path.dirname(CWD)

ROOT = CWD + "/resources/"


class Map:
    """Parent map class

    Parameters:
    -----------
    map_dir : str
        The absolute (or relative) path to the location where the .osm file is located

    Attributes:
    -----------
    osm_data            : dict
    croswalk_poly       : shapely.geometry.multipolygon.MultiPolygon
    intersection_poly   : shapely.geometry.multipolygon.MultiPolygon
    gap_poly            : shapely.geometry.multipolygon.MultiPolygon
    road_poly           : shapely.geometry.multipolygon.MultiPolygon
    sidewalk_poly       : shapely.geometry.multipolygon.MultiPolygon

    Functions:
    ----------
    plot_areas(alpha: float) -> matplotlib.pyplot.Axes
        Gets all the areas and plots the PolygonPatches in a matplotlib figure
        with the transparency alpha

    get_area(regex: str, tag_key: str) -> list
        given a regular expression and a key in the tags-list of the data-dict,
        this function returns the location of all nodes beloning to the region
        given by the regex

    """

    def __init__(self, map_dir: str = "SinD/Data/mapfile-Tianjin.osm"):
        self._map_dir = map_dir
        self.osm_data = self.__load_osm_data(ROOT + map_dir)
        self.__initialize_polygons() if re.findall("SinD", map_dir) else None

    def __load_osm_data(self, map_dir: str):
        osmhandler = OSMHandler()
        osmhandler.apply_file(map_dir)
        return osmhandler.osm_data


    def plot_single_data(
        self,
        pedestrian_data,
        map_overlay: bool = True,
        alpha: float = 0.2,
        padding_masks=None,
    ):
        # Create a single figure
        fig = plt.figure(figsize=(7 * 3, 6))  # Adjust figure size as needed

        # Add subplots on the same row. All three plots will now be in the same row.
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")  # First plot, 3D
        ax2 = fig.add_subplot(1, 3, 2)  # Second plot, 2D
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")  # Third plot, 3D

        # Plot map areas on ax2 if map_overlay is True
        if map_overlay:
            self.plot_areas(
                alpha=alpha, ax=ax2
            )  # Adjust this call according to your map object's API
        ax2.set_title("Pedestrian trajectories")

        for _id in pedestrian_data.keys():
            # Apply padding mask to filter out padded values
            data = pedestrian_data[_id]
            if padding_masks is not None:
                non_masked = padding_masks[_id]
                data = data[non_masked]

            x, y = np.array(data["x"]), np.array(data["y"])
            vx, vy = data["vx"], data["vy"]
            ax, ay = data["ax"], data["ay"]
            v = np.sqrt(np.array(vx).T ** 2 + np.array(vy).T ** 2)
            a = np.sqrt(np.array(ax).T ** 2 + np.array(ay).T ** 2)

            ax1.plot(x, y, zs=v, c="r"), ax1.set_title(
                "Velocity profile of trajectories"
            ), ax1.set_xlim(0, 30), ax1.set_ylim(0, 30), ax1.set_zlim(0, 5)
            ax1.set_xlabel("X"), ax1.set_ylabel("Y"), ax1.set_zlabel("V")
            ax2.plot(x, y, c="orange"), ax2.set_title("Pedestrian trajectories")
            ax3.plot(x, y, zs=a, c="r"), ax3.set_title(
                "Acceleration profile of trajectories"
            ), ax3.set_xlim(0, 30), ax3.set_ylim(0, 30), ax3.set_zlim(0, 5)
            ax3.set_xlabel("X"), ax3.set_ylabel("Y"), ax3.set_zlabel("A")
        plt.grid()
        plt.show()

    def plot_dataset(
        self,
        pedestrian_data: dict = {},
        color: str = "orange",
        map_overlay: bool = True,
        alpha: float = 0.2,
        alpha_trajectories: float = 1.0,
        size_points: int = 10,
        padding_masks=None,
        ax=None,
        title: str = "",
    ):
        show_plot = False
        if ax is None:
            show_plot = True
            ax = (
                self.plot_areas(alpha=alpha)
                if map_overlay
                else plt.figure(2).add_subplot()
            )
            ax.set_title(f"Pedestrian trajectories: {title}")

        for _id, data in pedestrian_data.items():
            # Apply padding mask to filter out padded values
            if padding_masks is not None:
                non_masked = padding_masks[_id]
                data = data[non_masked]

            x, y = np.array(data[:, 0]), np.array(data[:, 1])

            ax.plot(x, y, c=color, alpha=alpha_trajectories),

            # Mark the start and end points
            ax.scatter(x[0], y[0], c="green", s=size_points)
            ax.scatter(x[-1], y[-1], c="red", s=size_points)

        if show_plot:
            plt.grid()
            plt.show()

    def plot_dataset_color_clusters(
        self,
        pedestrian_data: dict = {},
        colors: list = [],
        clusters: list = [],
        map_overlay: bool = True,
        alpha: float = 0.2,
        alpha_trajectories: float = 1.0,
        size_points: int = 10,
        padding_masks=None,
        title: str = "",
    ):
        fig = plt.figure(figsize=(7*3, 6))  # Wider figure to accommodate three subplots
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        if map_overlay:
            self.plot_areas(alpha=alpha, ax=ax2)
        ax2.set_title(f"Pedestrian trajectories: {title}")

        used_labels = set()
        for (_id, data), color_id in zip(pedestrian_data.items(), clusters):
            if padding_masks is not None:
                non_masked = padding_masks[_id]
                data = data[non_masked]

            x, y = np.array(data[:, 0]), np.array(data[:, 1])
            vx, vy = np.array(data[:, 2]), np.array(data[:, 3])  # Assuming these are velocities
            ax, ay = np.array(data[:, 4]), np.array(data[:, 5])  # Assuming these are accelerations
            v = np.sqrt(vx**2 + vy**2)
            a = np.sqrt(ax**2 + ay**2)

            # Define label, plot data with or without adding it to the legend
            label = f"Cluster {color_id}"
            if label not in used_labels:
                ax2.plot(x, y, c=colors[color_id], alpha=alpha_trajectories, label=label)
                used_labels.add(label)
            else:
                ax2.plot(x, y, c=colors[color_id], alpha=alpha_trajectories)

            ax1.plot(x, y, zs=v, c=colors[color_id])
            ax3.plot(x, y, zs=a, c=colors[color_id])
            ax2.scatter(x[0], y[0], c="green", s=size_points)
            ax2.scatter(x[-1], y[-1], c="red", s=size_points)
            
        ax2.legend(title="Cluster")
        ax1.set_title("Velocity profile of trajectories")
        ax3.set_title("Acceleration profile of trajectories")

        ax1.set_xlabel("X"), ax1.set_ylabel("Y"), ax1.set_zlabel("Velocity")
        ax3.set_xlabel("X"), ax3.set_ylabel("Y"), ax3.set_zlabel("Acceleration")

        plt.grid()
        plt.show()

    def plot_dataset_color_clusters_all(
        self,
        pedestrian_data: dict = {},
        colors: list = [],
        clusters: list = [],
        map_overlay: bool = True,
        alpha: float = 0.2,
        alpha_trajectories: float = 1.0,
        size_points: int = 10,
        padding_masks=None,
        title: str = "",
    ):
        fig = plt.figure(figsize=(7*3, 6))  # Wider figure to accommodate three subplots
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        if map_overlay:
            self.plot_areas(alpha=alpha, ax=ax2)
        ax2.set_title(f"Pedestrian trajectories: {title}")

        used_labels = set()
        for (_id, data), color_id in zip(pedestrian_data.items(), clusters):
            if padding_masks is not None:
                non_masked = padding_masks[_id]
                data = data[non_masked]

            x, y = np.array(data[:, 0]), np.array(data[:, 1])
            vx, vy = np.array(data[:, 2]), np.array(data[:, 3])  # Assuming these are velocities
            ax, ay = np.array(data[:, 4]), np.array(data[:, 5])  # Assuming these are accelerations
            v = np.sqrt(vx**2 + vy**2)
            a = np.sqrt(ax**2 + ay**2)

            # Define label, plot data with or without adding it to the legend
            label = f"Cluster {color_id}"
            if label not in used_labels:
                ax2.plot(x, y, c=colors[color_id], alpha=alpha_trajectories, label=label)
                used_labels.add(label)
            else:
                ax2.plot(x, y, c=colors[color_id], alpha=alpha_trajectories)

            ax1.plot(x, y, zs=v, c=colors[color_id])
            ax3.plot(x, y, zs=a, c=colors[color_id])
            ax2.scatter(x[0], y[0], c="green", s=size_points)
            ax2.scatter(x[-1], y[-1], c="red", s=size_points)
            
        ax2.legend(title="Cluster", bbox_to_anchor=(1.23, 1), loc='upper left')
        ax1.set_title("Velocity profile of trajectories")
        ax3.set_title("Acceleration profile of trajectories")

        ax1.set_xlabel("X"), ax1.set_ylabel("Y"), ax1.set_zlabel("Velocity")
        ax3.set_xlabel("X"), ax3.set_ylabel("Y"), ax3.set_zlabel("Acceleration")

        plt.grid()
        plt.show()

    def get_area(self, regex: str = "crosswalk", tag_key: str = "name"):
        _ways, _nodes, _locs = [], [], []
        for _, values in self.osm_data["Relations"].items():
            tags = values["tags"]
            if tag_key not in tags.keys():
                break
            _found = re.findall(regex, tags[tag_key])
            if _found:
                _ways = [*_ways, *values["way_members"]]
        for _way in _ways:
            _nodes = [*_nodes, *self.osm_data["Ways"][_way]["nodes"]]
        for _node in _nodes:
            _locs.append(self.osm_data["Nodes"][_node])
        return _locs

    def __crosswalk_polygon(self):
        _points = self.get_area("crosswalk")
        crosswalk_shape = self.__get_exterior(_points)
        self.crosswalk_poly = (
            crosswalk_shape.difference(self.intersection_poly)
            if re.findall("Tianjin", ROOT + self._map_dir)
            else crosswalk_shape
        )

    def __intersection_polygon(self):
        _points = self.get_area("inter")
        self.intersection_poly = self.__get_exterior(_points)

    def __gap_polygon(self):
        _points = self.get_area("gap")
        self.gap_poly = self.__get_exterior(_points)

    def __road_and_sidewalk_polygon(self, sidewalk_size: float = 8.0):
        _points = self.get_area("")
        cross_poly = self.crosswalk_poly
        inter_poly = self.intersection_poly
        gap_poly = self.gap_poly
        road_poly = self.__get_exterior(_points)
        sidewalk_poly = (
            road_poly.buffer(sidewalk_size)
            .difference(road_poly)
            .intersection(road_poly.minimum_rotated_rectangle.buffer(-0.2))
        )
        self.road_poly, self.sidewalk_poly = (
            road_poly.difference(cross_poly)
            .difference(inter_poly)
            .difference(gap_poly),
            sidewalk_poly,
        )

    def __fix_poly_sizes(self):
        self.intersection_poly = self.intersection_poly.difference(
            self.crosswalk_poly.buffer(3)
        ).buffer(3)
        self.crosswalk_poly = (
            self.crosswalk_poly.buffer(3)
            .difference(self.intersection_poly)
            .difference(self.road_poly)
            .difference(self.sidewalk_poly)
            .difference(self.gap_poly)
        )

    def __initialize_polygons(self):
        self.__intersection_polygon(), self.__crosswalk_polygon(), self.__gap_polygon(), self.__road_and_sidewalk_polygon()
        self.__fix_poly_sizes()

    def __get_exterior(self, points: list, alpha: float = 0.4):
        return alphashape.alphashape(np.array(points), alpha=alpha)


class SinDMap(Map):
    """Map class for the SinD dataset"""

    def __init__(self, map_dir: str = "SinD/Data/mapfile-Tianjin.osm"):
        super().__init__(map_dir)

    def plot_areas(
        self,
        ax=None,
        highlight_areas: list = ["crosswalk", "sidewalk"],
        alpha: float = 0.08,
    ):
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(6.5, 4.13)
            fig.subplots_adjust(top=0.95, left=0.08, bottom=0.1, right=0.95)
            im = plt.imread(ROOT + "/intersection.jpg")
            ax.imshow(im, zorder=-1, alpha=0.5, extent=(-25, 60, -10, 40))
            # Optionally, set the axes limits to the dimensions of the image
            ax.set_xlim(-25, 60)
            ax.set_ylim(-10, 40)

        _points = self.get_area("")
        ax.scatter(*zip(*_points), alpha=0)  # To get bounds correct
        _attr = dir(self)
        _polys = [v for (_, v) in enumerate(_attr) if re.findall("poly$", v)]
        ["_".join([area, "poly"]) for area in highlight_areas]
        _ids = [
            _polys.index(i)
            for i in ["_".join([area, "poly"]) for area in highlight_areas]
        ]
        _colors = np.array(["r"] * len(_polys))
        _colors[_ids] = "green"
        _alphas = np.array([0.05] * len(_polys))
        _alphas[_ids] = alpha
        for i, _poly in enumerate(_polys):
            ax.add_patch(
                PolygonPatch(
                    eval(".".join(["self", _poly])), alpha=_alphas[i], color=_colors[i]
                )
            )

        return ax


class SVEAMap(Map):
    """Map class for SVEA-generated datasets"""

    def __init__(self, map_dir: str = "seven-eleven.osm"):
        super().__init__(map_dir)

    @staticmethod
    def rotate_about_point(image, angle, point):
        # Calculate the shifts
        shift_y, shift_x = np.array(image.shape[:2]) / 2.0 - np.array(point)

        # Shift the image to bring the point to the center
        shifted_image = shift(image, shift=[shift_y, shift_x, 0])

        # Rotate the image around the center
        rotated_image = rotate(shifted_image, angle, reshape=False)

        # Shift the image back to the original position
        final_image = shift(rotated_image, shift=[-shift_y, -shift_x, 0])

        return final_image

    def plot_areas(
        self,
        ax=None,
        highlight_areas: list = ["crosswalk", "sidewalk"],
        alpha: float = 0.08,
    ):
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(6.5, 4.13)
            fig.subplots_adjust(top=0.95, left=0.08, bottom=0.1, right=0.95)
            im = plt.imread(ROOT + "/seven-eleven3.png")
            # Define the new size (for example, half the original size)
            # Define the new size
            new_size = (161.156, 54.925)
            resized_image = resize(im, new_size, anti_aliasing=True)
            # Resize the image
            ax.imshow(resized_image, zorder=-1, alpha=0.5, extent=(-30, 131.156, -30, 24.925))
            # Optionally, set the axes limits to the dimensions of the image
            ax.set_xlim(50, 90)
            ax.set_ylim(-15, 15)

            # ax.set_xlim(-100, 100)
            # ax.set_ylim(-100, 100)

        # _points = self.get_area("")
        # _points
        # print(self.osm_data)
        # print(_points)
        # ax.scatter(*zip(*_points), alpha=0)  # To get bounds correct
        # center = LL2XYProjector().latlon2xy(59.3460774, 18.0716591)
        # x = [-100, 100]
        # y = [-100, 100]
        # ax.scatter(x, y, alpha=0)  # To get bounds correct
        # _attr = dir(self)
        # _polys = [v for (_, v) in enumerate(_attr) if re.findall("poly$", v)]
        # ["_".join([area, "poly"]) for area in highlight_areas]
        # _ids = [
        #     _polys.index(i)
        #     for i in ["_".join([area, "poly"]) for area in highlight_areas]
        # ]
        # _colors = np.array(["r"] * len(_polys))
        # _colors[_ids] = "green"
        # _alphas = np.array([0.05] * len(_polys))
        # _alphas[_ids] = alpha
        # for i, _poly in enumerate(_polys):
        #     ax.add_patch(
        #         PolygonPatch(
        #             eval(".".join(["self", _poly])), alpha=_alphas[i], color=_colors[i]
        #         )
        #     )

        return ax




class OSMHandler(osm.SimpleHandler):
    """OpenStreetMap handler that reads the nodes, ways and
    relations from a .osm-file
    """

    def __init__(self):
        """Format for osm_data
        osm_data = {
            Nodes: {
                _id: [x, y]
            },
            Ways" {
                _id: {
                    nodes: [_id1, ..., _idn],
                    tags: {_tags}
                }
            },
            Relations: {
                _id: {
                    way_members: [_id1, _id2],
                    tags: {_tags}
                }
            }
        }
        """
        osm.SimpleHandler.__init__(self)
        self.projector = LL2XYProjector()
        self.osm_data = {"Nodes": {}, "Ways": {}, "Relations": {}}

    def node(self, n):
        [x, y] = self.projector.latlon2xy(n.location.lat, n.location.lon)
        self.osm_data["Nodes"].update({n.id: [x, y]})

    def way(self, w):
        self.osm_data["Ways"].update(
            {w.id: {"nodes": [n.ref for n in w.nodes], "tags": dict(w.tags)}}
        )

    def relation(self, r):
        self.osm_data["Relations"].update(
            {
                r.id: {
                    "way_members": [m.ref for m in r.members if m.type == "w"],
                    "tags": dict(r.tags),
                }
            }
        )


class LL2XYProjector:
    """Projector class that projects longitude and latitude
    onto the xy-plane.

    Parameters:
    -----------
    lat_origin : float
        origin for latitude (default: 0)
    lon_origin : float
        origin for longitude (default: 0)

    Functions:
    ----------
    latlon2xy(lat: float, lon: float)
        converts latitude and longitude to xy-coordinates
        given the lat- and lon-origin
    """

    def __init__(self, lat_origin: float = 0, lon_origin: float = 0):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = (
            math.floor((lon_origin + 180.0) / 6) + 1
        )  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj="utm", ellps="WGS84", zone=self.zone, datum="WGS84")
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat: float, lon: float):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]
