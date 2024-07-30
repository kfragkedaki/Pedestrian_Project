from matplotlib.lines import Line2D
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd
import tqdm
import argparse
import sys

# Get the absolute path of the directory two levels up
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add that directory to sys.path
sys.path.append(project_dir)

from src.datasets.plot import SVEAMap, SinDMap
from src.reachability_analysis.labeling_oracle import LabelingOracleROSData, LabelingOracleSINDData, LABELS, LabelingOracleSVEAData
from src.reachability_analysis.operations import (
    visualize_zonotopes,
    input_zonotope,
    create_M_w,
    zonotope_area,
)
from src.reachability_analysis.reachability import LTI_reachability
from src.reachability_analysis.input_state import (
    create_io_state,
    separate_data_to_class,
    split_io_to_trajs,
    filter_paddings,
)
from src.reachability_analysis.zonotope import zonotope
from src.reachability_analysis.utils import load_data

from src.clustering.run import get_cluster, load_config

ROOT_PROJECT = os.getcwd()
ROOT_RESOURCES = os.getcwd() + "/resources"
DATADIR = "SinD/Data"
REVERSED_LABELS = {value: key for key, value in LABELS.items()}
COLORS = [
    [60 / 255, 159 / 255, 69 / 255, 0.4],  # Behavioral Zonotope
    [32 / 255, 102 / 255, 168 / 255, 0.4],  # Transformer-based Cluster Zonotope
    [0, 0, 0, 0.2],  # Cluster Zonotope
    [0.55, 0.14, 0.14, 0.4],  # Baseline Zonotope
    [63 / 255, 63 / 255, 63 / 255, 1],  # Past Trajectory
    [184 / 255, 184 / 255, 184 / 255, 1],  # Future Trajectory
    [0 / 255, 0 / 255, 0 / 255, 0.6],  # Current Position
    # [52/255 , 78/255 , 92/255 ], # Position
]

TEST_TRAJECTORIES = {
    'cross_illegal': 4,
    'not_cross': 26,
    'crossing_now': 3,
}

def reachability_for_specific_position_and_mode(
    pos: np.ndarray = np.array([-3.4, 28.3]),
    l: int = 1,
    vel: np.ndarray = np.array([1, 0]),
    _baseline: bool = True,
    _show_plot: bool = True,
    _ax: plt.Axes = None,
    _labels: list = None,
    _suppress_prints: bool = True,
    d: dict = None,
    sim: bool = False,
    sind: LabelingOracleSINDData = None,
    method: str = "",
    data_statistics: dict = None,
    clustering: bool = False
):
    """Get reachable set for a specific position, mode and starting velocity

    Parameters:
    -----------
    pos : np.ndarray
    c : int
    vel : np.ndarray
    _baseline : bool
    _show_plot : bool
    _ax : plt.Axes
    _labels : list
    _suppress_prints : bool
    _sind_ : LabelingOracleSINDData
    _d : np.ndarray
    """
    input_len = np.array(d[0]).shape[
        1
    ]  # (label_id, trajectory_id, input_len, features)
    reachability_sets_size = input_len - 1

    process_noise = 0.005
    v = vel
    c_z = pos
    z_w = zonotope(np.array([0, 0]), process_noise * np.ones(shape=(2, 1)))

    G_z = np.array([[2, 0, 1], [0, 2, 0.6]])
    z = zonotope(c_z, G_z)
    res = create_io_state(d, z, v, l, drop_equal=False, angle_filter=True, method=method, data_statistics=data_statistics, clustering=clustering)
    if res is None:
        return
    
    U, X_p, X_m, _ = res
    _, _, U_traj = split_io_to_trajs(
        X_p, X_m, U, threshold=5, dropped=False, N=reachability_sets_size
    )
    U_k = input_zonotope(U_traj, N=reachability_sets_size)
    M_w = create_M_w(U.shape[1], z_w, disable_progress_bar=sim)
    G_z = np.array([[0.5, 0, 0.25], [0, 0.5, 0.15]])
    z = zonotope(c_z, G_z)
    R_all = LTI_reachability(
        U,
        X_p,
        X_m,
        z,
        z_w,
        M_w,
        U_k,
        N=reachability_sets_size,
        disable_progress_bar=sim,
    )
    R = R_all[-1]
    R.color = COLORS[0]
    R_base_all = None

    if _baseline:
        G_z = np.array([[0.5, 0, 0.25], [0, 0.5, 0.15]])
        z = zonotope(c_z, G_z)
        res_baseline = create_io_state(
            d,
            z,
            v,
            list(set(d.keys())),
            drop_equal=False,
            angle_filter=False,
            method='baseline',
            data_statistics=data_statistics,
            clustering=False
        )
        if res_baseline is None:
            return
        U_all, X_p_all, X_m_all, _ = res_baseline
        _, _, U_all_traj = split_io_to_trajs(
            X_p_all,
            X_m_all,
            U_all,
            threshold=5,
            dropped=False,
            N=reachability_sets_size,
        )
        U_k_all = input_zonotope(U_all_traj, N=reachability_sets_size)
        M_w_base = create_M_w(U_all.shape[1], z_w, disable_progress_bar=sim)
        G_z = np.array([[0.5, 0, 0.25], [0, 0.5, 0.15]])
        z = zonotope(c_z, G_z)
        R_base_all = LTI_reachability(
            U_all,
            X_p_all,
            X_m_all,
            z,
            z_w,
            M_w_base,
            U_k_all,
            N=reachability_sets_size,
            disable_progress_bar=sim,
        )
        R_base = R_base_all[-1]
        R_base.color = COLORS[3]

    z.color = COLORS[6]
    _zonos = [R_base, R, z] if _baseline else [R, z]

    if not _suppress_prints:
        print("Area of zonotope: ", round(zonotope_area(R), 4), " m^2")
        if _baseline:
            print(
                "Area of (baseline) zonotope: ", round(zonotope_area(R_base), 4), " m^2"
            )

    if not _ax:
        _ax = sind.map.plot_areas()
    if sim:
        ax = visualize_zonotopes(_zonos, map=_ax, show=False, _labels=_labels)
    else:
        ax = None

    if _show_plot:
        plt.show()

    return ax, _zonos, R_all, R_base_all


def reachability_for_all_modes(
    pos: np.ndarray = np.array([-3.4, 28.3]),
    vel: np.ndarray = np.array([1, 0]),
    baseline: bool = False,
    config: dict = None,
    simulation: bool = False,
    test_cases: dict = None,
    trajectory: np.array = None,
    show_plot: bool = False,
    save_plot: str = None,
    load_data: bool = False,
    data_statistics: dict = None,
    title: str = ''
):
    """Reachability for all modes TODO update docstring

    Parameters:
    -----------
    pos : np.ndarray
    vel : np.ndarray
    _sind_ : SinD
    d_ : np.ndarray
    simulation : bool
    """
    _labels = list(test_cases.values())
    _z, _ids = [], []
    _b, _z_all = [], {}
    R_base_all_zonos = None
    z = None
    ax = None
    if data_statistics is not None: 
        data_statistics.update({'total_count':data_statistics['total_count']+1})
    else:
        print("Data statistics are not provided.")

    for i, (key, _label) in enumerate(test_cases.items()):
        method = _label.split(": ")[0]
        if method not in data_statistics.keys(): 
            data_statistics[method] = {"memory_constraint": 0, 'data_constraint': 0}

        if "co" in key:
            config["original_data"] = True
        else:
            config["original_data"] = False

        key = int(key.split("_")[1])
        _sind_, d_, _, mapping = get_data(
            _load=load_data, config=config, test_case=(key, _label)
        )
        _mode = mapping[key]

        clustering = False
        if "Label:" in _label:
            color_idx = 0
        elif "Cluster:" in _label:
            clustering = True
            if "T" in _label:
                color_idx = 1
            else:
                color_idx = 2

        if ax is None:
            ax = _sind_.map.plot_areas()

            if trajectory is not None:
                if trajectory.shape[0] > 1:
                    # test scenarios
                    x0, y0 = trajectory[0, :, 0], trajectory[0, :, 1]
                    ax.plot(x0, y0, c=COLORS[4], label="Past Trajectory", markersize=2)
                    x1, y1 = trajectory[1, :-1, 0], trajectory[1, :-1, 1]
                    ax.plot(
                        x1, y1, c=COLORS[5], label="Upcoming Trajectory", markersize=2
                    )
                else:
                    # In this case, we have the current location and want to predict the future
                    # calculate inclusion accuracy
                    x1, y1 = trajectory[0, :-1, 0], trajectory[0, :-1, 1]
                    ax.plot(
                        x1, y1, c=COLORS[5], label="Upcoming Trajectory", markersize=2
                    )

        # try:
        if baseline:
            baseline = True if not _b else False

        result_zonotopes = reachability_for_specific_position_and_mode(
            pos,
            _mode,
            vel,
            _baseline=baseline,
            _show_plot=False,
            _ax=ax,
            _suppress_prints=False,
            d=d_,
            sim=simulation,
            sind=_sind_,
            method=method,
            data_statistics=data_statistics,
            clustering=clustering
        )
        if result_zonotopes is None:
            return 
        
        _, _zonos, R_all, R_base_all = result_zonotopes
        if not baseline:
            R, z = _zonos
        else:
            R_base, R, z = _zonos
            _b.append(R_base)
            R_base_all_zonos = R_base_all

        R.color = COLORS[color_idx]
        _z.append(R)
        _ids.append(key)
        _z_all.update({_label: R_all})
        # except Exception as e:
        #     print(f"An error occurred: {e}, clustering: {clustering}")
        #     pass
    if z:
        _z.append(z)
        _labels.append("Current Position")
    if _b:
        _labels.insert(0, "Baseline")
        _z.insert(0, _b[0])
        _z_all.update({"baselines": R_base_all_zonos})

    visualize_zonotopes(
        _z, map=ax, save_plot=save_plot, show=show_plot, _labels=_labels, title=title
    )
    return _z, _labels, _b, _z_all


def scenario_func(
    trajectory: np.array,
    pos: np.ndarray,
    vel: np.ndarray,
    config: dict,
    test_cases: dict,
    baseline: bool = True,
    show_plot: bool = False,
    save_plot: str = None,
):
    """
    Run scenario for a specific mode.

    Parameters:
    -----------
    pos : np.ndarray
        Last position of the trajectory.
    vel : np.ndarray
        Last velocity of the trajectory.
    label : int
        Mode of the trajectory.

    Returns:
    --------
    z : zonotope
        Zonotope of the initial set.
    l : list
        Legends of the zonotopes.
    _z : list
        Zonotopes of all modes.
    _b : list
        Baseline zonotopes.
    _z_ : dict
        Zonotopes per mode.
    _b_ : list
        Baselines per mode.
    """
    _z_ = {}.fromkeys(test_cases.values())
    [_z_.update({i: []}) for i in _z_.keys()]
    if baseline:
        _z_["baselines"] = []
    _b_ = []
    data_statistics = {
        "baseline": {
            "memory_constraint": 0,
            'data_constraint': 0,
        },
        "total_count": 0,
    }

    for i in range(0, 1):
        save_plot_ = None
        if save_plot is not None:
            save_plot_ = save_plot + f"/{i}"

        res_zonotopes = reachability_for_all_modes(
            pos=pos,
            vel=vel,
            baseline=baseline,
            test_cases=test_cases,
            config=config,
            trajectory=trajectory,
            show_plot=show_plot,
            save_plot=save_plot_,
            load_data=True,
            data_statistics=data_statistics,
        )
        if res_zonotopes is None:
            return
        z, l, _b, _z = res_zonotopes

        for k, v in _z.items():
            if v:
                _z_[k].append(v[-1])

        if baseline:
            _b_.append(_b[-1])

    _f = open(ROOT_RESOURCES + f"/scenario.pkl", "wb")
    pickle.dump([z, l, _z, _b, _z_, _b_, data_statistics], _f)
    _f.close()


def get_data(
    _load: bool = False,
    _sind: LabelingOracleSINDData = None,
    config: dict = None,
    test_case: tuple = (0, "Label"),
):
    """Calculate the data separation to each class"""
    if not _sind and config['data_class'] == 'ros':
        _sind = LabelingOracleROSData(config)
    elif not _sind and config['data_class'] == 'sind':
        _sind = LabelingOracleSINDData(config)

    if os.path.exists(os.path.join(config["output_dir"], "clusters")):
        clusters_root = os.path.join(config["output_dir"], "clusters")
    else:
        clusters_root = ROOT_RESOURCES

    if "Cluster" in test_case[1]:
        # if _load:
        data = load_data(filename="/data_original.pkl", filepath=clusters_root)
        padded_batches = load_data(filename="/data_padding.pkl", filepath=clusters_root)
        labels = load_data(
            filename=f'/cluster_labels{"_original" if config["original_data"] else ""}.pkl',
            filepath=clusters_root,
        )
        # else:
        #     # cluster and get labels again
        #     run_clusters(config, load_embeddings=False, show_clusters=False)
        #     # test_labeling_oracle = get_test_config(config, test_name=test_name)
        #     c = get_cluster(config, data_oracle=test_labeling_oracle)
        #     data, padded_batches = load_data('data_original.pkl'), load_data('data_padding.pkl')
        #     labels = load_data(os.path.join(clusters_root, f'cluster_labels{"_original" if config["original_data"] else ""}.pkl.pkl'))
    else:
        if _load:
            data = load_data()
            padded_batches = load_data("sind_padding.pkl")
            labels = load_data("sind_labels.pkl")
        else:
            _sind.load_data()
            data, padded_batches = _sind.create_chunks(save_data=True)
            labels = _sind.labels(data, save_data=True)

    data = filter_paddings(data, padded_batches)
    labels = filter_paddings(labels, padded_batches)

    size = len(set(labels))
    mapping = {i: i for i in range(size)}
    if -1 in labels:
        mapping = {
            old_val: new_val for new_val, old_val in enumerate(sorted(set(labels)))
        }
        l = pd.DataFrame(labels)[0].map(mapping)
        labels = l.to_numpy()

    d_ = separate_data_to_class(data, labels, size=size)
    return _sind, d_, labels, mapping


def get_initial_conditions(data: np.ndarray, chunk: int = 0):
    point = -1  # last point
    x, y, vx, vy = (
        data[chunk][point][0],
        data[chunk][point][1],
        data[chunk][point][2],
        data[chunk][point][3],
    )
    pos = np.array([x, y])
    v = np.array([vx, vy])
    return pos, v


def run_scenario(
    trajectory: np.ndarray,
    config: dict,
    labels: list,
    show_path: bool = True,
    baseline: bool = True,
    show_plot: bool = False,
    save_plot: str = None,
):
    pos, v = get_initial_conditions(trajectory)
    if not show_path:
        trajectory = None
    scenario_func(
        trajectory,
        pos,
        v,
        config,
        labels,
        baseline=baseline,
        show_plot=show_plot,
        save_plot=save_plot,
    )


def get_test_label(test_labeling_oracle, chunk:int = 0):
    trajectory, _ = test_labeling_oracle.create_chunks(save_data=False)
    label = test_labeling_oracle.labels(trajectory[chunk:chunk+1], save_data=False)

    return trajectory[chunk:chunk+2], label[0]


def get_test_config(config: dict, test_name: str = "", original_data: bool = False):
    config_test = config.copy()
    config_test["data_dir"] = ROOT_RESOURCES + f"/test/{test_name}"
    config_test["original_data"] = original_data
    test_labeling_oracle = LabelingOracleSINDData(config_test)
    return test_labeling_oracle


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run clustering script with arguments."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="experiments",
        help="Folder that includes the trained models.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="SINDDataset_pretrained_2024-04-27_00-11-45_KIP",
        help="The model to create clusters for.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=2,
        help="The index number, which indicates the right path to the models' folder.",
    )
    parser.add_argument(
        "--index_data",
        type=int,
        default=0,
        help="The index number, which indicates the right path to the data folder.",
    )
    parser.add_argument(
        "--original_data",
        type=bool,
        default=False,
        help="If the original data should be used for clustering.",
    )

    # Parse the arguments
    args = parser.parse_args()

    config = load_config(
        folder=args.folder,
        model_file=args.model_file,
        index=args.index,
        index_data=args.index_data,
        original_data=args.original_data,
    )

    for test_name, chunk in TEST_TRAJECTORIES.items():
        test_labeling_oracle = get_test_config(config, test_name=test_name)
        test_labeling_oracle.load_data()

        trajectory, l = get_test_label(test_labeling_oracle, chunk=chunk)
        c, distance_c = get_cluster(config, data_oracle=test_labeling_oracle, chunk=chunk)

        test_labeling_oracle.config['original_data'] = True
        co, distance_co = get_cluster(config, test_labeling_oracle, chunk=chunk)

        if distance_co > 3 or distance_c > 3:
            print("No cluster found that is close enough")
            continue

        test_cases = {
            f"l_{l}": f"Label: {REVERSED_LABELS[l]}",
            f"c_{c}": f"T-b Cluster: {c}",
            f'co_{co}': f'Cluster: {co}'
        }

        print(test_cases)
        run_scenario(
            trajectory=trajectory, config=config, labels=test_cases, show_plot=True, baseline=True
        )
