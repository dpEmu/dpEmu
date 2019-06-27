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
