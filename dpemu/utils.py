from datetime import datetime
from os.path import join
from pathlib import Path


def generate_unique_path(folder_name, extension, prefix=None):
    """[summary]

    [extended_summary]

    Args:
        folder_name ([type]): [description]
        extension ([type]): [description]
        prefix ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    root_folder = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if prefix:
        return join(root_folder, "{}/{}_{}.{}".format(folder_name, prefix, timestamp, extension))
    return join(root_folder, "{}/{}.{}".format(folder_name, timestamp, extension))


def get_project_root():
    """[summary]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    return Path(__file__).resolve().parents[1]


def split_df_by_model(df):
    """[summary]

    [extended_summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    dfs = []
    for model_name, df_ in df.groupby("model_name"):
        df_ = df_.dropna(axis=1, how="all")
        df_ = df_.drop("model_name", axis=1)
        df_ = df_.reset_index(drop=True)
        df_.name = model_name
        dfs.append(df_)
    return dfs


def filter_optimized_results(df, err_param_name, score_name, is_higher_score_better):
    """[summary]

    [extended_summary]

    Args:
        df ([type]): [description]
        err_param_name ([type]): [description]
        score_name ([type]): [description]
        is_higher_score_better (bool): [description]

    Returns:
        [type]: [description]
    """
    if is_higher_score_better:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmax()].reset_index(drop=True)
    else:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmin()].reset_index(drop=True)
    df_.name = df.name
    return df_
