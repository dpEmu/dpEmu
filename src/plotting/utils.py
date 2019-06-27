import matplotlib.pyplot as plt

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
            ax.set_ylim([0, 1])
            ax.legend()

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
        ax.set_title(err_param_name + "=" + str(df[err_param_name][i]))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cbar = fig.colorbar(sc, ax=axs, boundaries=np.arange(11) - 0.5, ticks=np.arange(10), use_gridspec=True)
    if label_names:
        cbar.ax.yaxis.set_ticklabels(label_names)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)
