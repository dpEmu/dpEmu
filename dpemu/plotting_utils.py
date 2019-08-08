import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphviz import Digraph
from matplotlib.colors import LinearSegmentedColormap

from .problemgenerator.filters import Filter
from .utils import generate_unique_path, split_df_by_model, filter_optimized_results

pd.set_option("display.expand_frame_repr", False)


def visualize_scores(df, score_names, is_higher_score_better, err_param_name, title, log=False):
    """Plots the wanted scores for all distinct models that were used.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        score_names (list): A list of strings which are the names of the scores for which we want to create a plot.
        is_higher_score_better (list): A list of booleans for each score type: True means that a higher score
            is better and False means a lower score is better.
        err_param_name (str): The error whose distinct values are going to be used on the x-axis.
        title (str): The title of the plot.
        log (bool, optional): A bool telling whether a logarithmic scale should be used on x-axis or not.
            Defaults to False.
    """

    dfs = split_df_by_model(df)

    n_scores = len(score_names)
    fig, axs = plt.subplots(1, n_scores, figsize=(n_scores * 5, 4), squeeze=False)
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


def visualize_best_model_params(
            df, model_name, model_params, score_names, is_higher_score_better,
            err_param_name, title, x_log=False, y_log=False
        ):
    """Plots the best model parameters for distinct error values.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        model_name (str): The name of the model for which we want to plot the best parameters.
        model_params (list): A list of strings which are the names of the params of the model we want to plot.
        score_names (list): A list of strings which are the names of the scores for which we want to create a plot.
        is_higher_score_better (list): A list of booleans for each score type: True means that a higher score
            is better and False means a lower score is better.
        err_param_name (str): The error whose distinct values are going to be used on the x-axis.
         title (str): The title of the plot.
        x_log (bool, optional): A bool telling whether a logarithmic scale should be used on x-axis or not.
            Defaults to False.
        y_log (bool, optional): A bool telling whether a logarithmic scale should be used on y-axis or not.
            Defaults to False.
    """
    dfs = split_df_by_model(df)

    for model_param in model_params:
        plt.figure()
        ax = plt.subplot(111)
        for i, _ in enumerate(score_names):
            for df_ in dfs:
                if df_.name != model_name:
                    continue
                df_ = filter_optimized_results(df_, err_param_name, score_names[i], is_higher_score_better[i])
                if x_log and y_log:
                    plt.loglog(df_[err_param_name], df_[model_param], label=score_names[i])
                elif x_log:
                    plt.semilogx(df_[err_param_name], df_[model_param], label=score_names[i])
                elif y_log:
                    plt.semilogy(df_[err_param_name], df_[model_param], label=score_names[i])
                else:
                    plt.plot(df_[err_param_name], df_[model_param], label=score_names[i])
                    ax.set_xlim([df_[err_param_name].min(), df_[err_param_name].max()])
                ax.set_xlabel(err_param_name)
                ax.set_ylabel(model_param)
                plt.legend(fontsize="small")

        plt.subplots_adjust(wspace=.25)
        plt.suptitle(title + " (" + model_name + ")")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path_to_plot = generate_unique_path("out", "png")
        plt.savefig(path_to_plot)


def visualize_classes(df, label_names, err_param_name, reduced_data_column, labels_column, cmap, title):
    """This function visualizes the classes as 2-dimensional plots for different error parameter values.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        label_names (list): A list containing the names of the labels.
        err_param_name (str): The name of the error parameter whose different values are used for plots.
        reduced_data_column (str): The name of the column that contains the reduced data.
        labels_column (str): The name of the column that contains the labels for each element.
        cmap (str): The name of the color map used for coloring the plot.
        title (str): The title of the plot.
    """

    def get_lims(data):
        """Returns the limits of the plot.

        Args:
            data (list): A list of 2-dimensional data points.

        Returns:
            float, float, float, float: minimum x, maximum x, minimum y, maximum y.
        """
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = df.groupby(err_param_name).first().reset_index()
    labels = df[labels_column][0]

    n_col = math.ceil(df.shape[0] / 2)
    fig, axs = plt.subplots(2, n_col, figsize=(2.5 * n_col + 1, 5), constrained_layout=True)
    for i, ax in enumerate(axs.ravel()):
        if i >= df.shape[0]:
            ax.set_xticks([])
            ax.set_yticks([])
            plt.box(False)
            continue
        reduced_data = df[reduced_data_column][i]
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
    """Creates an interactive plot for each different value of the given error type.

    The data points in the plots can be clicked to activate a given function.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        err_param_name (str): The name of error parameter based on which the data is grouped by.
        data (obj): The original data that was given to the runner module.
        scatter_cmap (str): The color map for the scatter plot
        reduced_data_column (str): The name of the column containing the reduced data
        on_click (function): A function used for interactive plotting.
            When a data point is clicked, the function is given the original and modified elements as its parameters.
    """

    def get_lims(data):
        """Returns the limits of the plot.

        Args:
            data (list): A list of 2-dimensional data points.

        Returns:
            float, float, float, float: minimum x, maximum x, minimum y, maximum y.
        """
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
    """Creates a confusion matrix which can be made interactive if wanted.

    Args:
        df_ (DataFrame): The original dataframe returned by the runner.
        cm (list): An integer matrix describing the number of elements in each category of the confusion matrix.
        row (int): The row of the dataframe used for this matrix.
        label_names (list): A list of strings containing the names of the labels.
        title (str): The title of the confusion matrix visualization.
        labels_column (str): The name of the column containing the real labels.
        predicted_labels_column (str): The name of the column containing the predicted labels.
        on_click (function, optional): If this parameter is passed to the function, then the interactive mode.
            will be set on and clicking an element causes the event listener to call this function.
            The function should take three parameters: an element, a real label and a predicted label. Defaults to None.
    """
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
        """This class describes a combination of a plot and an event listener.

        It is required so that the event listeners refer to a correct row of data
        and the references exist after the function is run.
        """

        def __init__(self, row, fig, df_, cm_values, on_click):
            """
            Args:
                row (int): The row of the original dataframe whose data the matrix uses.
                fig (Figure): The figure to which the confusion matrix is plotted.
                df_ (DataFrame): The original dataframe returned by the runner.
                cm_values (list): A matrix of lists containing the elements of each category of the confusion matrix.
                on_click (function): A function to be called after a cell is clicked.
            """
            self.row = row
            self.fig = fig
            self.cid = None
            self.df_ = df_
            self.cm_values = cm_values
            if on_click:
                self.on_click = on_click
                self.cid = fig.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            """This function passes an element from the clicked category to the on_click function.

            This function is called by the event listener.

            Args:
                event (Event): The button press event which activated the event listener.
            """
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
                                 predictions_col, on_click=None):
    """Generates confusion matrices for each error parameter combination and model.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        label_names (list): A list containing the names of the labels.
        score_name (str): The name of the score type used for filtering the best results.
        is_higher_score_better (bool): If true, then a higher value of score is better and vice versa.
        err_param_name (str): The name of the error parameter whose different values the matrices use.
        labels_col (str): The name of the column containing the real labels.
        predictions_col (str): The name of the column containing the predicted labels.
        on_click (function, optional): If this parameter is passed to the function, then the interactive mode
            will be set on and clicking an element causes the event listener to call this function.
            The function should take three parameters: an element, a real label and a predicted label. Defaults to None.
    """
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
                on_click
            )


def visualize_error_generator(root_node, view=True):
    """Generates a directed graph describing the error generation tree and filters.

    root_node.generate_error() needs to be called before calling this function,
    because otherwise Filters may have incorrect or missing parameter values
    in the graph.

    Args:
        root_node (Node): The root node of the error generation tree.
        view (bool, optional): If view is True then the error generation tree graph is displayed to user
            in addition to saving it to a file. If False then it's only saved to file in DOT graph
            description language. Defaults to True.

    Returns:
        str: File path to the saved DOT graph description file.
    """

    dot = Digraph()
    index = 0
    max_param_value_length = 40

    def describe_filter(ftr, parent_index, edge_label):
        """Describes a filter as a dot node.

        Args:
            ftr (Filter): The filter to be described.
            parent_index (int): The index of the parent node or filter.
            edge_label (str): The label of the edge.
        """
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
        """Describes a node as a dot node.

        Args:
            node (Node): [The node to be described.
            parent_index (int): The index of the parent node.
        """
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
    """Prints the dataframe row by row excluding the unwanted columns.

    Args:
        df (pandas.DataFrame): The dataframe returned by the runner.
        dropped_columns (list, optional): List of the column names we do not want to be printed. Defaults to [].
    """
    dropped_columns.extend(["interactive_err_data"])

    dfs = split_df_by_model(df)
    for df_ in dfs:
        print(df_.name)
        print(df_.drop(columns=[col for col in dropped_columns if col in df_]))
