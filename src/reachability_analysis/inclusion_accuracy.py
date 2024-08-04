import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from copy import deepcopy
from src.reachability_analysis.operations import is_inside, zonotope_area
from src.reachability_analysis.labeling_oracle import LABELS
from src.reachability_analysis.simulation import (
    reachability_for_all_modes,
    get_test_label,
    get_test_config,
)
from src.clustering.run import get_cluster, load_config

import pandas as pd
import matplotlib.ticker as tck
import argparse


REVERSED_LABELS = {value: key for key, value in LABELS.items()}

ROOT_TEST = os.getcwd() + "/resources/test/"
DATASET = "8_02_1"
RA_PATH = "/SinD/reachable_sets.pkl"
RAB_PATH = "/SinD/reachable_base_sets.pkl"


def load_data_for_simulation(
    name: str = "Ped_smoothed_tracks.csv", input_len: int = 90, load_data: bool = False
):
    """Load the dataset in such way that it can be simulated
    with appropriate frame appearances from pedestrians

    Parameters:
    -----------
    name : str (default = 'Ped_smoothed_tracks.csv')
    """
    _path = "/".join([ROOT_TEST, DATASET, name])
    _data = pd.read_csv(_path)
    _last_frame = _data["frame_id"].max() + 1
    ped_data_for_RA = {}
    pedestrian_data = {}.fromkeys(list(range(0, _last_frame)))
    [pedestrian_data.update({i: {}}) for i in pedestrian_data.keys()]
    for _id in _data["track_id"].unique():
        ped = _data.loc[_data["track_id"] == _id]
        _, _f, x, y, vx, vy, ax, ay = (
            ped["track_id"],
            ped["frame_id"],
            ped["x"],
            ped["y"],
            ped["vx"],
            ped["vy"],
            ped["ax"],
            ped["ay"],
        )
        ped_data_for_RA.update(
            {
                _id: {
                    "frame_id": _f,
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "ax": ax,
                    "ay": ay,
                }
            }
        )
    _data_chunks_for_RA = generate_input_for_sim(
        ped_data_for_RA, _last_frame, input_len, load_data
    )
    for _det in _data.values:
        _id, _f, _, _, x, y, vx, vy, ax, ay = _det
        if _id != "P2":  # Specific for 8_02_1
            pedestrian_data[_f].update(
                {_id: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}
            )
    return pedestrian_data, _data_chunks_for_RA, _last_frame


def generate_input_for_sim(
    data: dict, _last_frame: int, input_len: int = 90, load_data: bool = False
):
    """Generate the trajectory chunks for reachability analysis

    Parameters:
    -----------
    data : dict
        Dictionary of pedestrian data
    _last_frame : int
        The last frame in the dataset
    input_len : int
        The length of each chunk
    """
    if not load_data:
        _concat_data = {}.fromkeys(list(range(0, _last_frame)))
        [_concat_data.update({i: {}}) for i in _concat_data.keys()]
        for _j, _data in tqdm(data.items(), desc="Retrieving input"):
            if _j != "P2":  # Specific for 8_02_1
                _f, x, y, vx, vy, ax, ay = (
                    _data["frame_id"],
                    _data["x"],
                    _data["y"],
                    _data["vx"],
                    _data["vy"],
                    _data["ax"],
                    _data["ay"],
                )
                num_chunks = (
                    len(x) // input_len
                )  # Calculate how many full chunks can be made
                for _i in range(num_chunks-1):
                    start_past_idx = _i * input_len
                    end_past_idx = start_past_idx + input_len
                    _x_past, _y_past = np.array(x.iloc[start_past_idx:end_past_idx]), np.array(
                        y.iloc[start_past_idx:end_past_idx]
                    )
                    _vx_past, _vy_past = np.array(vx.iloc[start_past_idx:end_past_idx]), np.array(
                        vy.iloc[start_past_idx:end_past_idx]
                    )
                    _ax_past, _ay_past = np.array(ax.iloc[start_past_idx:end_past_idx]), np.array(
                        ay.iloc[start_past_idx:end_past_idx]
                    )

                    start_future_idx = end_past_idx
                    end_future_idx = start_future_idx + input_len
                    _x_future, _y_future = np.array(x.iloc[start_future_idx:end_future_idx]), np.array(
                        y.iloc[start_future_idx:end_future_idx]
                    )
                    _vx_future, _vy_future = np.array(vx.iloc[start_future_idx:end_future_idx]), np.array(
                        vy.iloc[start_future_idx:end_future_idx]
                    )
                    _ax_future, _ay_future = np.array(ax.iloc[start_future_idx:end_future_idx]), np.array(
                        ay.iloc[start_future_idx:end_future_idx]
                    )
                    _frame = _f.values[end_past_idx]

                    if (
                        _frame in _concat_data
                    ):  # Make sure the frame index exists in the dictionary
                        _concat_data[_frame].update(
                            {
                                _j: {
                                    "x": np.array([_x_past, _x_future]),
                                    "y": np.array([_y_past, _y_future]),
                                    "vx": np.array([_vx_past, _vx_future]),
                                    "vy": np.array([_vy_past, _vy_future]),
                                    "ax": np.array([_ax_past, _ax_future]),
                                    "ay": np.array([_ay_past, _ay_future]),
                                }
                            }
                        )
        return _concat_data
    else:
        _file = open(ROOT_TEST + "sim_dict.json", "rb")
        _new_data = pickle.load(_file)
        _file.close()
    return _new_data


def _simulation(
    load_data: bool = True,
    checkpoint: int = 5,
    frames: int = None,
    _baseline: bool = True,
    config: dict = None,
    original_data: bool = False,
):
    """Simulating the DATASET using the "true" mode (from the labeling oracle)

    Parameters:
    -----------
    input_len : int (default = 90)
    load_data : bool (default = True)
    load_calc_d_data : bool (default = True)
    checkpoint : int (default = 10)
    frames : int (default = None)
    all_modes : bool (default = False)
    """
    input_len = config["data_chunk_len"]
    _data, _RA_data, _last_frame = load_data_for_simulation(
        input_len=input_len, load_data=False
    )
    test_labeling_oracle = get_test_config(
        config, test_name=DATASET, original_data=False
    )
    if original_data:
        test_labeling_oracle_original = get_test_config(
            config, test_name=DATASET, original_data=True
        )

    data_statistics = {
        "baseline": {
            "memory_constraint": 0,
            'data_constraint': 0,
        },
        "total_count": 0,
        "distances_failed": {
            "Transformer-encoded": 0,
            "Non-encoded": 0
        }
    }

    if not load_data:
        RA_l = {}.fromkeys(list(range(0, _last_frame)))
        [RA_l.update({i: {}}) for i in RA_l.keys()]
        RA_c = deepcopy(RA_l)
        RA_co = deepcopy(RA_l)
        RA_b = deepcopy(RA_l)
        for frame in tqdm(
            _data.keys() if not frames else range(0, frames),
            desc="Simulating " + DATASET,
        ):
            _RA = _RA_data[frame]
            for _ped_id, state in _data[frame].items():
                if _ped_id in _RA:
                    print(f"Pedestrian: {_ped_id}")
                    historical_trajectory = {key: value[0, :] for key, value in _RA[_ped_id].items()}
                    future_trajectory = {key: value[1, :] for key, value in _RA[_ped_id].items()}

                    pos = np.array([historical_trajectory["x"][-1], historical_trajectory["y"][-1]])
                    vel = np.array([historical_trajectory["vx"][-1], historical_trajectory["vy"][-1]])
                    past = pd.DataFrame(historical_trajectory)
                    past["track_id"] = 0

                    future = pd.DataFrame(future_trajectory)
                    future["track_id"] = 1
                    trajectory = pd.concat([past, future])

                    test_labeling_oracle.all_df = trajectory.set_index("track_id")
                    test_labeling_oracle.feature_df = test_labeling_oracle.all_df[test_labeling_oracle.feature_names]
                    test_labeling_oracle.all_IDs =  test_labeling_oracle.all_df.index.unique()

                    plotted_path, l = get_test_label(test_labeling_oracle)
                    c, distance_c = get_cluster(config, test_labeling_oracle)
                    test_cases = {
                        f"l_{l}": f"Labeling: {REVERSED_LABELS[l]}",
                        f"c_{c}": f"Transformer-encoded: {c}",
                    }

                    if distance_c > 3:
                        data_statistics["distances_failed"].update({"Transformer-encoded": data_statistics["distances_failed"]["Transformer-encoded"]+1})

                    if original_data:
                        test_labeling_oracle_original.all_df = trajectory.set_index("track_id")
                        test_labeling_oracle_original.feature_df = test_labeling_oracle_original.all_df[
                            test_labeling_oracle_original.feature_names]
                        test_labeling_oracle_original.all_IDs = test_labeling_oracle_original.all_df.index.unique()

                        co, distance_co = get_cluster(
                            config, test_labeling_oracle_original
                        )
                        test_cases[f"co_{co}"] = f"Non-encoded: {co}"

                        if distance_co > 3:
                            print("Non-encoded Trajectory Clustering: No cluster found with relatively low distance. Distance: ", distance_co)
                            print(f"Data Statistics: {data_statistics['distances_failed']}")
                            data_statistics["distances_failed"].update(
                                {"Non-encoded": data_statistics["distances_failed"]["Non-encoded"] + 1})
                            continue

                    if distance_c > 3:
                        print("Transformer-encoded Trajectory Clustering: No cluster found with relatively low distance. Distance: ", distance_c)
                        print(f"Data Statistics: {data_statistics['distances_failed']}")
                        continue

                    print(f"Test_cases: {test_cases}")
                    res_zontopes = reachability_for_all_modes(
                        pos,
                        vel,
                        baseline=_baseline,
                        config=config,
                        test_cases=test_cases,
                        show_plot=True,
                        trajectory=plotted_path,
                        load_data=True,
                        data_statistics=data_statistics,
                        title=f"Pedestrian {_ped_id}, Frame: {frame}, pos {pos}"
                    )
                    if res_zontopes is None:
                        continue
                    
                    _, _, _, _z_all = res_zontopes

                    if f"l_{l}" in test_cases: 
                        RA_l[frame].update(
                        {_ped_id: {"zonotopes": _z_all[test_cases[f"l_{l}"]], "id": l}}
                    )
                    if f"c_{c}" in test_cases:
                        RA_c[frame].update(
                            {_ped_id: {"zonotopes": _z_all[test_cases[f"c_{c}"]], "id": c}}
                        )
                    if original_data:
                        RA_co[frame].update(
                            {
                                _ped_id: {
                                    "zonotopes": _z_all[test_cases[f"co_{co}"]],
                                    "id": co,
                                }
                            }
                        )

                    if _baseline:
                        RA_b[frame].update({_ped_id: _z_all["baselines"]})

            if frame % checkpoint and frame != 0:
                _f = open(ROOT_TEST + DATASET + "_accuracy.pkl", "wb")
                pickle.dump([RA_l, RA_c, RA_co, RA_b, data_statistics, (frame, _last_frame)], _f)
                _f.close()

        _f = open(ROOT_TEST + DATASET + "_accuracy.pkl", "wb")
        pickle.dump([RA_l, RA_c, RA_co, RA_b, data_statistics, (frame, _last_frame)], _f)
        _f.close()
    else:
        _f = open(ROOT_TEST + DATASET + "_accuracy.pkl", "rb")
        RA_l, RA_c, RA_co, RA_b, data_statistics, _frames = pickle.load(_f)
        _f.close()

    RA_c_volume, i_c_volume = 0, 0
    RA_l_volume, i_l_volume = 0, 0
    RA_b_volume, i_b_volume = 0, 0
    RA_co_volume, i_co_volume = 0, 0

    RA_b_acc = np.array([0] * input_len)
    RA_l_acc = np.array([0] * input_len)
    RA_c_acc = np.array([0] * input_len)
    RA_co_acc = np.array([0] * input_len)
    i_l = np.array([0] * input_len)
    i_c = np.array([0] * input_len)
    i_b = np.array([0] * input_len)
    i_co = np.array([0] * input_len)
    for frame in tqdm(RA_l.keys(), desc="Calculating accuracy for " + DATASET):
        _RA = _RA_data[frame]
        for _ped_id, state in _data[frame].items():
            if _ped_id in _RA:
                if RA_l[frame]:
                    _z_l_all = RA_l[frame][_ped_id]["zonotopes"]
                if RA_c[frame]:
                    _z_c_all = RA_c[frame][_ped_id]["zonotopes"]
                if RA_co[frame]:
                    _z_co_all = RA_co[frame][_ped_id]["zonotopes"]

                if RA_b[frame] and _baseline:
                    _b = RA_b[frame][_ped_id]

                for k in range(0, input_len - 1):
                    state_k = _data[frame + k][_ped_id]
                    pos_k = np.array([state_k["x"], state_k["y"]])
                    try:
                        # T-f Clsuters
                        if RA_c[frame]:
                            zono = _z_c_all[k]
                            if zonotope_area(zono) < 1000:
                                # exclude outliers
                                _inside = int(is_inside(zono, pos_k))
                                RA_c_acc[k] += _inside
                                i_c[k] += 1
                                if _inside and k==input_len - 2:
                                    # last zonotope
                                    RA_c_volume += zonotope_area(zono)
                                    i_c_volume += 1

                        # Original Clsuters
                        if RA_co[frame]:
                            zono = _z_co_all[k]
                            if zonotope_area(zono)<1000:
                                # exclude outliers
                                _inside = int(is_inside(zono, pos_k))
                                RA_co_acc[k] += _inside
                                i_co[k] += 1
                                if _inside and k==input_len - 2:
                                    # last zonotope
                                    RA_co_volume += zonotope_area(zono)
                                    i_co_volume += 1

                        # Labels
                        if RA_l[frame]:
                            zono = _z_l_all[k]
                            if zonotope_area(zono)<1000:
                                # exclude outliers
                                _inside = int(is_inside(zono, pos_k))
                                RA_l_acc[k] += _inside
                                i_l[k] += 1
                                if _inside and k==input_len - 2:
                                    # last zonotope
                                    RA_l_volume += zonotope_area(zono)
                                    i_l_volume += 1

                        # Baseline
                        if RA_b[frame] and _baseline:
                            if zonotope_area(_b[k])<1000:
                                # exclude outliers
                                _inside = int(
                                    is_inside(_b[k], pos_k)
                                )  # Does not have all zonotopes
                                RA_b_acc[k] += _inside
                                i_b[k] += 1
                                if _inside and k==input_len - 2:
                                    # last zonotope
                                    RA_b_volume += zonotope_area(_b[k])
                                    i_b_volume += 1

                    except Exception as err:
                        print("Error Label", err)
                        pass

    ids_l = np.where(i_l == 0)[0]
    ids_c = np.where(i_c == 0)[0]
    ids_b = np.where(i_b == 0)[0]
    ids_co = np.where(i_co == 0)[0]
    RA_c_acc, _i_c = RA_c_acc[0 : ids_c[0]], i_c[0 : ids_c[0]]
    RA_l_acc, _i_l = RA_l_acc[0 : ids_l[0]], i_l[0 : ids_l[0]]
    RA_b_acc, _i_b = RA_b_acc[0 : ids_b[0]], i_b[0 : ids_b[0]]
    RA_co_acc, _i_co = RA_co_acc[0 : ids_co[0]], i_co[0 : ids_co[0]]

    print(f"Data Statistics: {data_statistics}")
    _f_stats = open(ROOT_TEST + "state_inclusion_acc_statistics.pkl", "wb")
    pickle.dump(data_statistics, _f_stats)

    if i_b_volume !=0 :
        print(f"Average Volumes: Baseline: {RA_b_volume/i_b_volume}, Labeling:{RA_l_volume/i_l_volume}, \
         Non-encoded Trajectory Clustering:{RA_co_volume/i_co_volume}, Transformer-encoded Trajectory Clustering:{RA_c_volume/i_c_volume}")
        _f_volumes =  open(ROOT_TEST + "state_inclusion_acc_volumes.pkl", "wb")
        pickle.dump({"baseline": [RA_b_volume, i_b_volume], "labeling": [RA_l_volume, i_l_volume], "clustering": [RA_co_volume, i_co_volume], "transformer": [RA_c_volume, i_c_volume]}, _f_volumes)

    print(
        f"Labeling Acurracy: {RA_l_acc / _i_l}, T-f Clustering Accuracy: {RA_c_acc / _i_c}, Baseline Accuracy: {RA_b_acc/_i_b}"
    )
    _f_l = open(ROOT_TEST + "state_inclusion_acc_label.pkl", "wb")
    _f_c = open(ROOT_TEST + "state_inclusion_acc_clsuter.pkl", "wb")
    _f_b = open(ROOT_TEST + "state_inclusion_acc_baseline.pkl", "wb")

    pickle.dump(RA_c_acc / _i_c * 100, _f_c)
    pickle.dump(RA_l_acc / _i_l * 100, _f_l)
    pickle.dump(RA_b_acc / _i_b * 100, _f_b)

    _f_c.close()
    _f_l.close()
    _f_b.close()

    if original_data:
        print(f"Original Clustering Accuracy: {RA_co_acc / _i_co}")
        _f_co = open(ROOT_TEST + "state_inclusion_acc_cluster_original.pkl", "wb")
        pickle.dump(RA_co_acc / _i_co * 100, _f_co)
        _f_co.close()


COLORS = [
    [60 / 255, 159 / 255, 69 / 255, 1],  # Behavioral Zonotope
    [32 / 255, 102 / 255, 168 / 255, 1],  # Transformer-based Cluster Zonotope
    [0.55, 0.14, 0.14, 1],  # Baseline Zonotope
    # [239/ 255, 201/ 255, 88/ 255, 1] # Cluster Zonotope
    [0, 0, 0, 1],  # Cluster Zonotope
]


def visualize_state_inclusion_acc(
    baseline: bool = True,
    convergence: bool = True,
    side: str = "right",
    original_data: bool = False,
):
    # plt.rcParams.update({'font.size': 14})
    if baseline:
        _f_b = open(ROOT_TEST + "state_inclusion_acc_baseline.pkl", "rb")
        RA_b_acc = pickle.load(_f_b)
        _f_b.close()

    _f_l = open(ROOT_TEST + "state_inclusion_acc_label.pkl", "rb")
    _f_c = open(ROOT_TEST + "state_inclusion_acc_clsuter.pkl", "rb")

    if original_data:
        _f_co = open(ROOT_TEST + "state_inclusion_acc_cluster_original.pkl", "rb")
        RA_co_acc = pickle.load(_f_co)
        _f_co.close()

    RA_l_acc = pickle.load(_f_l)
    RA_c_acc = pickle.load(_f_c)
    _f_l.close()
    _f_c.close()

    fig, ax = plt.subplots()
    fig.set_size_inches(8 / 1.5, 4.2 / 1.5)
    fig.subplots_adjust(top=0.96, left=0.090, bottom=0.165, right=0.93)

    _x = np.array(list(range(2, len(RA_l_acc) + 1))) / 10
    if baseline:
        plt.plot(_x, RA_b_acc[1:], "-.", color=COLORS[2], label="Baseline")
    ax.plot(_x, RA_l_acc[1:], "--", color=COLORS[0], label="Labeling")

    if original_data:
        ax.plot(_x, RA_co_acc[1:], ":", color=COLORS[3], label="Non-encoded", lw=1.8)

    ax.plot(_x, RA_c_acc[1:], "-", color=COLORS[1], label="Transformer-encoded")


    ax.set_ylim([0, 110]), ax.set_xlim([0, 5])
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.minorticks_on()

    ax.set_ylabel("Accuracy [%]", fontweight="bold", fontsize="14")
    ax.set_xlabel("Time horizon, N [s]", fontweight="bold", fontsize="14")
    ax.legend()

    ax.grid(which="major")
    ax.grid(which="minor", ls="--", linewidth=0.33)

    if convergence:
        ax1 = ax.twinx()
        ax1.set_yticks([86], ["86%"])
        ax1.tick_params(axis="y", colors=COLORS[3], labelsize=10) # Cluster
        ax1.grid(alpha=0.6)
        ax1.set_ylim([0, 110])
        ax1.yaxis.set_ticks_position(side)

        ax2 = ax.twinx()
        ax2.set_yticks([94], ["94%"])
        ax2.tick_params(axis="y", colors=COLORS[1], labelsize=10) # T-b
        ax2.grid(alpha=0.6)
        ax2.set_ylim([0, 110])
        ax2.yaxis.set_ticks_position(side)
        ax3 = ax.twinx()
        ax3.set_yticks([98], ["98%"])
        ax3.tick_params(axis="y", colors=COLORS[2], labelsize=10) # Baseline
        ax3.grid(alpha=0.6)
        ax3.set_ylim([0, 110])
        ax3.yaxis.set_ticks_position(side)

        ax4 = ax.twinx()
        ax4.set_yticks([90], ["90%"])
        ax4.tick_params(axis="y", colors=COLORS[0], labelsize=10) # Baseline
        ax4.grid(alpha=0.6)
        ax4.set_ylim([0, 110])
        ax4.yaxis.set_ticks_position(side)

        # Adjust label position by offsetting
        for label in ax3.get_yticklabels():
            label.set_verticalalignment('bottom')  # Adjusts vertical alignment to move label up

        # Adjust label position by offsetting
        for label in ax2.get_yticklabels():
            label.set_verticalalignment('baseline')  # Adjusts vertical alignment to move label up

        # Adjust label position by offsetting
        for label in ax1.get_yticklabels():
            label.set_verticalalignment('top')  # Adjusts vertical alignment to move label up

    plt.savefig(ROOT_TEST + "accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


def get_state_inclusion_acc(config, original_data=False):
    """Code to reproduce the state inclusion accuracy

    NOTE: This might take forever to compute. To tackle
    this, try decreasing the value of the 'frames' argument.
    """
    _simulation(
        load_data=True, config=config, _baseline=True, original_data=original_data
    )
    visualize_state_inclusion_acc(
         baseline=True, convergence=True, original_data=original_data
    )


if __name__ == "__main__":
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Run clustering script with arguments."
    )
    parser.add_argument(
        "--original_data",
        type=bool,
        default=False,
        help="If the original data should be used for clustering.",
    )
    args = parser.parse_args()

    print("Original_data: ", args.original_data)
    get_state_inclusion_acc(config, original_data=args.original_data)
