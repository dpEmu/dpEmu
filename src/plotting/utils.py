import math
import random

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from matplotlib.colors import LinearSegmentedColormap

from src.problemgenerator.filters import Filter
from src.utils import generate_unique_path, split_df_by_model, filter_optimized_results


def visualize_scores(df, score_names, is_higher_score_better, err_param_name, title, log=False):
    dfs = split_df_by_model(df)

    n_scores = len(score_names)
    fig, axs = plt.subplots(1, n_scores, figsize=(n_scores * 4, 4), squeeze=False)
    for i, ax in enumerate(axs.ravel()):
        for df_ in dfs:
            df_ = filter_optimized_results(df_, err_param_name, score_names[i], is_higher_score_better[i])
            if log:
                ax.semilogx(df_[err_param_name], df_[score_names[i]], label=df_.name)
            else:
                ax.plot(df_[err_param_name], df_[score_names[i]], label=df_.name)
                ax.set_xlim([df_[err_param_name].min(), df_[err_param_name].max()])
            ax.set_xlabel(err_param_name)
            ax.set_ylabel(score_names[i])
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
    fig, axs = plt.subplots(2, n_col, figsize=(2.5 * n_col + 1, 5), constrained_layout=True)
    for i, ax in enumerate(axs.ravel()):
        if i >= df.shape[0]:
            ax.set_xticks([])
            ax.set_yticks([])
            plt.box(False)
            continue
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
    n_unique = np.unique(labels).shape[0]
    cbar = fig.colorbar(sc, ax=axs, boundaries=np.arange(n_unique + 1) - 0.5, ticks=np.arange(n_unique),
                        use_gridspec=True, aspect=50)
    if label_names:
        cbar.ax.yaxis.set_ticklabels(label_names)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize_interactive_plot(df, err_param_name, data, scatter_cmap, reduced_data_column, on_click):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = df.groupby(err_param_name).first().reset_index()
    labels = df["labels"][0]

    # plot the data of each error parameter combination
    for i, _ in enumerate(df[reduced_data_column]):
        reduced_data = df[reduced_data_column][i]
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
            def __init__(self, i, fig, reduced_T, on_click):
                self.i = i
                self.fig = fig
                self.cid = self.fig.canvas.mpl_connect('pick_event', self)
                self.reduced_T = reduced_T
                self.on_click = on_click

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
                original = data[closest]
                modified = df["interactive_err_data"][self.i][closest]

                self.on_click(original, modified)

        Plot(i, fig, reduced_T, on_click)


def visualize_confusion_matrix(df_, cm, row, label_names, title, labels_column, predicted_labels_column, on_click=None):
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
            label = label_names[df_[labels_column][row][index]]
            predicted_label = label_names[df_[predicted_labels_column][row][index]]
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


def visualize_confusion_matrices(df, label_names, score_name, is_higher_score_better, err_param_name, labels_col,
                                 predictions_col, interactive):
    dfs = split_df_by_model(df)
    for df_ in dfs:
        df_ = filter_optimized_results(df_, err_param_name, score_name, is_higher_score_better)
        for i in range(df_.shape[0]):
            visualize_confusion_matrix(
                df_,
                df_["confusion_matrix"][i],
                i,
                label_names,
                f"{df_.name} confusion matrix ({err_param_name}={round(df_[err_param_name][i], 3)})",
                labels_col,
                predictions_col,
                interactive
            )


def visualize_error_generator(root_node, view=True):
    """Generates a directed graph describing the error generation tree and filters.

    root_node.generate_error() needs to be called before calling this function,
    because otherwise Filters may have incorrect or missing parameter values
    in the graph
    """

    dot = Digraph()
    index = 0
    max_param_value_length = 40

    def describe_filter(ftr, parent_index, edge_label):
        nonlocal index
        index += 1
        my_index = index

        # construct the label of the node
        label = "< " + str(ftr.__class__.__name__)
        for key in vars(ftr):
            if key[-3:] == "_id" or key == "shape":
                continue
            value = ftr.__dict__[key]
            if isinstance(value, Filter):
                continue
            value = str(value)
            if len(value) > max_param_value_length:
                value = value[:max_param_value_length] + "..."
            label += "<BR /><FONT POINT-SIZE='8'>" + str(key) + ": " + str(value) + "</FONT>"
        label += " >"

        # add a node and an edge to the digraph
        dot.node(str(my_index),
                 label=label,
                 _attributes={'shape': 'box'})
        dot.edge(str(parent_index), str(my_index), label=edge_label, _attributes={"fontsize": "8"})

        # describe all child filters
        for key in vars(ftr):
            value = ftr.__dict__[key]
            if isinstance(value, Filter):
                describe_filter(value, my_index, key)

    def describe(node, parent_index):
        nonlocal index
        index += 1
        my_index = index
        dot.node(str(my_index), label="< " + str(node.__class__.__name__) + " >")
        if parent_index:
            dot.edge(str(parent_index), str(my_index))
        for child in node.children:
            describe(child, my_index)
        for ftr in node.filters:
            describe_filter(ftr, my_index, "")

    describe(root_node, None)

    path_to_graph = generate_unique_path("out", "gv")
    dot.render(path_to_graph, view=view)
    return path_to_graph


def print_results(df, dropped_columns=[]):
    dropped_columns.extend(["interactive_err_data"])

    dfs = split_df_by_model(df)
    for df_ in dfs:
        print(df_.name)
        print(df_.drop(columns=[col for col in dropped_columns if col in df_]))
