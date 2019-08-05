import shlex
import subprocess

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP


def run_ml_module_using_cli(cline):
    """[summary]

    [extended_summary]

    Args:
        cline ([type]): [description]
    """
    proc = subprocess.Popen(shlex.split(cline), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    lines = []
    while True:
        line = proc.stdout.readline().rstrip()
        if line or proc.poll() is not None:
            print(line)
            lines.append(line)
        if not line and proc.poll() is not None:
            break
    return "\n".join(lines)


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
