import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.utils import generate_unique_path


def visualize_scores(dfs, score_names, err_param_name, title):
    n_scores = len(score_names)
    fig, axs = plt.subplots(1, n_scores, figsize=(n_scores * 4, 4))
    for i, ax in enumerate(axs.ravel()):
        for df in dfs:
            df_ = df.groupby(err_param_name, sort=False)[score_names[i]].max()
            ax.plot(df_.index, df_, label=df.name)
            ax.set_xlabel(err_param_name)
            ax.set_ylabel(score_names[i])
            ax.set_xlim([0, df_.index.max()])
            ax.legend(fontsize="small")

    fig.subplots_adjust(wspace=.25)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize_classes(dfs, label_names, err_param_name, title):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = dfs[0].groupby(err_param_name).first().reset_index()
    labels = df["labels"][0]

    n_col = math.ceil(df.shape[0] / 2)
    fig, axs = plt.subplots(2, n_col, figsize=(2.5 * n_col + 1, 5))
    for i, ax in enumerate(axs.ravel()):
        reduced_data = df["reduced_data"][i]
        x_min, x_max, y_min, y_max = get_lims(reduced_data)
        sc = ax.scatter(*reduced_data.T, c=labels, cmap="tab10", marker=".", s=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        err_param_val = round(df[err_param_name][i], 3)
        ax.set_title(err_param_name + "=" + str(err_param_val))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cbar = fig.colorbar(sc, ax=axs, boundaries=np.arange(11) - 0.5, ticks=np.arange(10), use_gridspec=True)
    if label_names:
        cbar.ax.yaxis.set_ticklabels(label_names)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize_interactive(dfs, err_param_name, data, scatter_cmap, image_cmap, shape=None):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    # Guess the shape of the data using the assumption that the shape of the images is a square
    if not shape:
        rgb = data[0].shape[-1] == 3
        rgba = data[0].shape[-1] == 4
        prod = 1
        for x in data[0].shape:
            prod *= x
        if rgb:
            prod //= 3
        elif rgba:
            prod //= 4
        sqrt = 0
        while (sqrt + 1) * (sqrt + 1) <= prod:
            sqrt += 1
        if sqrt * sqrt != prod:
            print("Unable to guess the shape of the data. Please specify it in visualize_interactive()'s parameters.")
            return
        if rgb:
            shape = (sqrt, sqrt, 3)
        elif rgba:
            shape = (sqrt, sqrt, 4)
        else:
            shape = (sqrt, sqrt)
        print("The program assumes that the shape of the data is", shape)
        print("If this is incorrect, please specify the shape in visualize_interactive()'s parameters.")

    df = dfs[0].groupby(err_param_name).first().reset_index()
    labels = df["labels"][0]

    # plot the data of each error parameter combination
    for i, _ in enumerate(df["reduced_data"]):
        reduced_data = df["reduced_data"][i]
        x_min, x_max, y_min, y_max = get_lims(reduced_data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(reduced_data.T[0], reduced_data.T[1], c=labels, cmap=scatter_cmap, marker=".", s=40, picker=True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        err_param_val = round(df[err_param_name][i], 3)
        ax.set_title(err_param_name + "=" + str(err_param_val))
        ax.set_xticks([])
        ax.set_yticks([])

        reduced_T = reduced_data.T

        # without creating a class the plots would use wrong values of i
        class Plot:
            def __init__(self, i, fig, reduced_T):
                self.i = i
                self.fig = fig
                self.cid = self.fig.canvas.mpl_connect('pick_event', self)
                self.reduced_T = reduced_T

            def __call__(self, event):
                if len(event.ind) == 0:
                    return False
                mevent = event.mouseevent
                closest = event.ind[0]

                def dist(x0, y0, x1, y1):
                    return (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)

                # find closest data point
                for elem in event.ind:
                    best_dist = dist(self.reduced_T[0][elem], self.reduced_T[1][elem], mevent.xdata, mevent.ydata)
                    new_dist = dist(self.reduced_T[0][closest], self.reduced_T[1][closest], mevent.xdata, mevent.ydata)
                    if best_dist > new_dist:
                        closest = elem

                # get original and modified data points
                elem = data[closest].reshape(shape)
                modified = df['err_test_data'][self.i][closest].reshape(shape)

                # create a figure and draw the images
                fg, axs = plt.subplots(1, 2)
                axs[0].matshow(elem, cmap=image_cmap)
                axs[0].axis('off')
                axs[1].matshow(modified, cmap=image_cmap)
                axs[1].axis('off')
                fg.show()

        Plot(i, fig, reduced_T)


def visualize_confusion_matrix(cm, label_names, title):
    # Draw image of confusion matrix
    color_map = LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0.2, 0.2, 1)], 256)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, color_map)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    min_val = np.amin(cm)
    max_val = np.amax(cm)
    break_point = (max_val + min_val) / 2

    plt.ylabel("true label")
    plt.xlabel("predicted label")

    # Loop over data dimensions and create text annotations.
    for i in range(n):
        for j in range(n):
            col = (1, 1, 1)
            if cm[i, j] <= break_point:
                col = (0, 0, 0)
            ax.text(j, i, cm[i, j], ha="center", va="center", color=col, fontsize=12)

    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot, bbox_inches="tight")


def visualize_confusion_matrices(dfs, label_names, score_name, err_param_name):
    for df in dfs:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmax()].reset_index(drop=True)
        for i in range(df_.shape[0]):
            visualize_confusion_matrix(
                df_["confusion_matrix"][i],
                label_names,
                f"{df.name} confusion matrix ({err_param_name}={round(df_[err_param_name][i], 3)})",
            )


def print_dfs(dfs, dropped_columns=[]):
    for df in dfs:
        print(df.name)
        print(df.drop(columns=dropped_columns))
