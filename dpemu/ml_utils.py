from os.path import isfile
from shlex import split
from subprocess import Popen, PIPE, STDOUT
from sys import stdout

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP

from dpemu.utils import get_project_root


def run_ml_module_using_cli(cline, show_stdout=True):
    """[summary]

    [extended_summary]

    Args:
        show_stdout:
        cline ([type]): [description]
    """
    if show_stdout:
        proc = Popen(split(cline), bufsize=0, stdout=PIPE, stderr=STDOUT, universal_newlines=True,
                     cwd=get_project_root())
    else:
        proc = Popen(split(cline), bufsize=0, stdout=PIPE, universal_newlines=True, cwd=get_project_root())
    chars = []
    while True:
        char = proc.stdout.read(1)
        if not char and proc.poll() is not None:
            print()
            break
        if char and show_stdout:
            stdout.write(char)
            stdout.flush()
        if char:
            chars.append(char)
    return "".join(chars)


def reduce_dimensions(data, random_state, target_dim=2):
    """[summary]

    [extended_summary]

    Args:
        data ([type]): [description]
        random_state ([type]): [description]
        target_dim (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    jl_limit = johnson_lindenstrauss_min_dim(n_samples=data.shape[0], eps=.3)
    pca_limit = 30

    if data.shape[1] > jl_limit and data.shape[1] > pca_limit:
        data = SparseRandomProjection(n_components=jl_limit, random_state=random_state).fit_transform(data)

    if data.shape[1] > pca_limit:
        data = PCA(n_components=pca_limit, random_state=random_state).fit_transform(data)

    return UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=random_state).fit_transform(data)


def reduce_dimensions_sparse(data, random_state, target_dim=2):
    """[summary]

    [extended_summary]

    Args:
        data ([type]): [description]
        random_state ([type]): [description]
        target_dim (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    svd_limit = 30

    if data.shape[1] > svd_limit:
        data = TruncatedSVD(n_components=svd_limit, random_state=random_state).fit_transform(data)

    return UMAP(n_components=target_dim, random_state=random_state).fit_transform(data)


def load_yolov3():
    path_to_yolov3_weights = f"{get_project_root()}/tmp/yolov3-spp_best.weights"
    if not isfile(path_to_yolov3_weights):
        Popen(["./scripts/get_yolov3.sh"], cwd=get_project_root()).wait()
    return path_to_yolov3_weights, f"{get_project_root()}/tmp/yolov3-spp.cfg"
