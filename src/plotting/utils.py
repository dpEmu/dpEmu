import random

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.utils import generate_unique_path, split_df_by_model, filter_optimized_results


def visualize_scores(df, score_names, err_param_name, title):
    dfs = split_df_by_model(df)

    n_scores = len(score_names)
    fig, axs = plt.subplots(1, n_scores, figsize=(n_scores * 4, 4))
    for i, ax in enumerate(axs.ravel()):
        for df_ in dfs:
            df_ = filter_optimized_results(df_, err_param_name, score_names[i])
            ax.plot(df_[err_param_name], df_[score_names[i]], label=df_.name)
            ax.set_xlabel(err_param_name)
            ax.set_ylabel(score_names[i])
            ax.set_xlim([0, df_[err_param_name].max()])
            ax.legend(fontsize="small")

    fig.subplots_adjust(wspace=.25)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize_classes(df, label_names, err_param_name, reduced_data_name, labels_name, cmap, title):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = df.groupby(err_param_name).first().reset_index()
    labels = df[labels_name][0]

    n_col = math.ceil(df.shape[0] / 2)
    fig, axs = plt.subplots(2, n_col, figsize=(2.5 * n_col + 1, 5))
    for i, ax in enumerate(axs.ravel()):
        reduced_data = df[reduced_data_name][i]
        x_min, x_max, y_min, y_max = get_lims(reduced_data)
        sc = ax.scatter(*reduced_data.T, c=labels, cmap=cmap, marker=".", s=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        err_param_val = round(df[err_param_name][i], 3)
        ax.set_title(err_param_name + "=" + str(err_param_val))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    n_unique = np.unique(labels).shape[0]
    cbar = fig.colorbar(sc, ax=axs, boundaries=np.arange(n_unique + 1) - 0.5, ticks=np.arange(n_unique),
                        use_gridspec=True)
    if label_names:
        cbar.ax.yaxis.set_ticklabels(label_names)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize_interactive_plot(df, err_param_name, data, scatter_cmap, image_cmap, shape=None):
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

    df = df.groupby(err_param_name).first().reset_index()
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
                modified = df["interactive_err_data"][self.i][closest].reshape(shape)

                # create a figure and draw the images
                fg, axs = plt.subplots(1, 2)
                axs[0].matshow(elem, cmap=image_cmap)
                axs[0].axis('off')
                axs[1].matshow(modified, cmap=image_cmap)
                axs[1].axis('off')
                fg.show()

        Plot(i, fig, reduced_T)


def visualize_confusion_matrix(df_, cm, row, label_names, title, on_click=None):
    # Draw image of confusion matrix
    color_map = LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0.2, 0.2, 1)], 256)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, color_map)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    cm_values = {}
    if on_click:
        for label in label_names:
            cm_values[label] = {}
            for label_prediction in label_names:
                cm_values[label][label_prediction] = []
        for index, _ in enumerate(df_["interactive_err_data"][row]):
            label = label_names[df_["test_labels"][row][index]]
            predicted_label = label_names[df_["predicted_test_labels"][row][index]]
            cm_values[label][predicted_label].append(index)

    class Plot:
        def __init__(self, row, fig, df_, cm_values, on_click):
            self.row = row
            self.fig = fig
            self.cid = None
            self.df_ = df_
            self.cm_values = cm_values
            if on_click:
                self.on_click = on_click
                self.cid = fig.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.xdata and event.ydata:
                x, y = int(round(event.xdata)), int(round(event.ydata))
                label = label_names[y]
                predicted = label_names[x]
                if self.cm_values[label][predicted]:
                    index = random.choice(self.cm_values[label][predicted])
                    self.on_click(self.df_["interactive_err_data"][self.row][index], label, predicted)

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

    Plot(row, fig, df_, cm_values, on_click)

    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot, bbox_inches="tight")


def visualize_confusion_matrices(df, label_names, score_name, err_param_name, interactive):
    dfs = split_df_by_model(df)
    for df_ in dfs:
        df_ = filter_optimized_results(df_, err_param_name, score_name)
        for i in range(df_.shape[0]):
            visualize_confusion_matrix(
                df_,
                df_["confusion_matrix"][i],
                i,
                label_names,
                f"{df_.name} confusion matrix ({err_param_name}={round(df_[err_param_name][i], 3)})",
                interactive
            )


def print_results(df, dropped_columns=[]):
    dfs = split_df_by_model(df)

    dropped_columns.extend(["interactive_err_data"])
    dropped_columns = [dropped_column for dropped_column in dropped_columns if dropped_column in df]

    for df_ in dfs:
        print(df_.name)
        print(df_.drop(columns=dropped_columns))
