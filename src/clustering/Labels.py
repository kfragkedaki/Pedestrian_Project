from src.reachability_analysis.labeling_oracle import LabelingOracleSINDData, LABELS
from src.clustering.run import run_clusters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

REVERSED_LABELS = {value: key for key, value in LABELS.items()}
COLORS = [
    "#EF3D59",
    "#E17A47",
    "#EFC958",
    "#4AB19D",
    "#344E5C",
    "#A6206A",
    "#568EA6",
    "#A2D4AB",
    "#5A5050",
]


def get_color_palette(num_data):
    cmap = cm.get_cmap("hsv")
    COLOR_PALETTE = [cmap(i / num_data) for i in range(num_data)]

    return COLOR_PALETTE


def label_trajectories(
    data_original: np.array,
    padding_masks: np.array,
    clusters: pd.DataFrame,
    labels: pd.DataFrame,
):
    unique_labels = np.unique(labels)
    trajectories = {label: {} for label in unique_labels}
    padding_per_label = {label: {} for label in unique_labels}
    clusters_per_label = {label: {} for label in unique_labels}

    for label in unique_labels:
        feature_df = data_original[labels == label]
        padding_per_label[label] = padding_masks[labels == label]
        clusters_per_label[label] = clusters[labels == label]
        for i, (index, row) in enumerate(feature_df.iterrows()):
            trajectories[label][i] = row.values.reshape(-1, 6)

    return trajectories, padding_per_label, clusters_per_label


def plot_trajectories_per_label(
    labeling_oracle, trajectories, padding_masked, COLOR_PALETTE=COLORS
):
    for key, label in REVERSED_LABELS.items():
        labeling_oracle.map.plot_dataset(
            pedestrian_data=trajectories[key],
            color=COLOR_PALETTE[key],
            title=f"{label}",
            alpha_trajectories=0.85,
            size_points=5,
            padding_masks=padding_masked[key],
        )


def plot_trajectories_per_label_color_clusters(
    labeling_oracle, trajectories, padding_masked, clusters, COLOR_PALETTE=COLORS
):
    key = 3
    labeling_oracle.map.plot_dataset_color_clusters(
        pedestrian_data=trajectories[key],
        colors=COLOR_PALETTE,
        clusters=clusters[key],
        title=f"{REVERSED_LABELS[key]}",
        alpha_trajectories=0.9,
        size_points=0,
        padding_masks=padding_masked[key],
    )


# Plotting Radar Chart
def plot_radar_chart(ax, values, categories, title, color):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = values.tolist() + values.tolist()[:1]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.plot(angles, values, color=color, linewidth=1, linestyle="solid", label=title)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_title(title, size=22, color="red", y=1.1)


def plot_clusters_distribution_per_label(
    labels, data_original, clusters, COLOR_PALETTE=COLORS
):
    data_pool_original = pd.DataFrame(
        np.mean(data_original, axis=1), columns=["x", "y", "vx", "vy", "ax", "ay"]
    )
    clusters_df_ = pd.DataFrame(clusters, columns=["cluster"])
    labels_df = pd.DataFrame(labels, columns=["label"])
    labels_df["label"] = labels_df["label"].map(REVERSED_LABELS)

    cluster_features = pd.concat([labels_df, clusters_df_, data_pool_original], axis=1)
    category_counts = (
        cluster_features.groupby(["cluster", "label"]).size().unstack(fill_value=0)
    )
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)

    num_clusters = len(category_proportions)
    cols = min(7, num_clusters)
    rows = (num_clusters + cols - 1) // cols
    fig, axs = plt.subplots(
        figsize=(cols * 5, rows * 5),
        nrows=rows,
        ncols=cols,
        subplot_kw=dict(polar=True),
    )
    plt.rcParams.update({"font.size": 16})
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    # Plot each cluster's radar chart
    for i, (idx, row) in enumerate(category_proportions.iterrows()):
        plot_radar_chart(
            axs[i], row, row.index, f"Cluster {idx}", color=COLOR_PALETTE[i]
        )

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_dual_tsne_3d(data_cluster1, data_cluster2, n_components=3, figsize=(14, 6)):
    """
    Applies t-SNE and plots the results in 3D for two datasets side by side.

    :param data1: First dataset (original data, their clusters)
    :param data2: Second dataset (embeddings, their clusters)
    :param clusters: Cluster labels for each sample
    :param n_components: Number of PCA components to compute
    :param figsize: Size of the figure for plotting
    """

    data1, clusters1 = data_cluster1
    data2, clusters2 = data_cluster2
    if len(data1.shape) > 2:
        data1 = data1.reshape(data1.shape[0], -1)
    if len(data2.shape) > 2:
        data2 = data2.reshape(data2.shape[0], -1)
    # Initialize PCA and reduce dimensions
    tsne = TSNE(
        n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42
    )
    data1_transformed = tsne.fit_transform(data1)
    data2_transformed = tsne.fit_transform(data2)

    # Prepare the plot
    fig = plt.figure(figsize=figsize)
    COLOR_PALETTE = get_color_palette(max(len(set(clusters1)), len(set(clusters2))))

    # Plotting for the first dataset
    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        data1_transformed[:, 0],
        data1_transformed[:, 1],
        data1_transformed[:, 2],
        c=clusters1,
        cmap=ListedColormap(COLOR_PALETTE),
        edgecolor="k",
    )
    legend1 = ax1.legend(
        *scatter1.legend_elements(), loc="upper right", title="Clusters"
    )
    ax1.add_artist(legend1)
    ax1.set_title("t-SNE of Original Data")

    # Plotting for the second dataset
    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter(
        data2_transformed[:, 0],
        data2_transformed[:, 1],
        data2_transformed[:, 2],
        c=clusters2,
        cmap=ListedColormap(COLOR_PALETTE),
        edgecolor="k",
    )
    legend2 = ax2.legend(
        *scatter2.legend_elements(), loc="upper right", title="Clusters"
    )
    ax2.add_artist(legend2)
    ax2.set_title("t-SNE of Embeddings")

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_dual_pca_3d(
    data_cluster1,
    data_cluster2,
    n_components=3,
    figsize=(14, 6),
    file: str = "pca_plot",
):
    """
    Applies PCA and plots the results in 3D for two datasets side by side.

    :param data1: First dataset (original data, their clusters)
    :param data2: Second dataset (embeddings, their clusters)
    :param clusters: Cluster labels for each sample
    :param n_components: Number of PCA components to compute
    :param figsize: Size of the figure for plotting
    """

    data1, clusters1 = data_cluster1
    data2, clusters2 = data_cluster2
    if len(data1.shape) > 2:
        data1 = data1.reshape(data1.shape[0], -1)
    if len(data2.shape) > 2:
        data2 = data2.reshape(data2.shape[0], -1)
    # Initialize PCA and reduce dimensions
    pca = PCA(n_components=n_components)
    data1_transformed = pca.fit_transform(data1)
    data2_transformed = pca.fit_transform(data2)

    # Prepare the plot
    fig = plt.figure(figsize=figsize)
    COLOR_PALETTE = get_color_palette(max(len(set(clusters1)), len(set(clusters2))))

    # Plotting for the first dataset
    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        data1_transformed[:, 0],
        data1_transformed[:, 1],
        data1_transformed[:, 2],
        c=clusters1,
        cmap=ListedColormap(COLOR_PALETTE),
        edgecolor="k",
    )
    # legend1 = ax1.legend(*scatter1.legend_elements(), loc='upper right', title="Clusters")
    # ax1.add_artist(legend1)
    ax1.set_title("PCA of Original Data", fontweight="bold", fontsize=14)

    # Plotting for the second dataset
    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter(
        data2_transformed[:, 0],
        data2_transformed[:, 1],
        data2_transformed[:, 2],
        c=clusters2,
        cmap=ListedColormap(COLOR_PALETTE),
        edgecolor="k",
    )
    # legend2 = ax2.legend(*scatter2.legend_elements(), loc='upper right', title="Clusters")
    # ax2.add_artist(legend2)
    ax2.set_title("PCA of Embeddings", fontweight="bold", fontsize=14)
    plt.subplots_adjust(wspace=0)
    # Display the plot
    # plt.tight_layout()
    plt.savefig(f"{file}.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_labels(
    config: dict,
    remove_noise: bool = True,
    show_clusters: bool = False,
    show_labels: bool = False,
    show_labeled_clusters: bool = True,
):
    clusters, embeddings, target, padding_masks = run_clusters(
        config, load_embeddings=True, load_clusters=True, show_clusters=show_clusters
    )
    COLOR_PALETTE = get_color_palette(len(set(clusters)))

    labeling_oracle = LabelingOracleSINDData(config)
    labels = labeling_oracle.labels(target)
    # Flatten the sequence_length and features dimensions
    df_target = pd.DataFrame(
        target.reshape(target.shape[0], -1)
    )  # flatten data merge first two dimensions

    if remove_noise:
        df_target = df_target[clusters != -1]
        labels = labels[clusters != -1]
        padding_masks = padding_masks[clusters != -1]
        clusters = clusters[clusters != -1]

    trajectories, padding_per_label, clusters_per_label = label_trajectories(
        df_target, padding_masks, clusters, labels
    )
    if show_labels:
        plot_trajectories_per_label(
            labeling_oracle=labeling_oracle,
            trajectories=trajectories,
            padding_masked=padding_per_label,
        )
    if show_labeled_clusters:
        plot_trajectories_per_label_color_clusters(
            labeling_oracle,
            trajectories=trajectories,
            padding_masked=padding_per_label,
            clusters=clusters_per_label,
            COLOR_PALETTE=COLOR_PALETTE,
        )
        plot_clusters_distribution_per_label(
            labels=labels,
            data_original=target,
            clusters=clusters,
            COLOR_PALETTE=COLOR_PALETTE,
        )
