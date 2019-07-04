import subprocess

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP


def run_ml_script(cline):
    _run_script(cline.split())


def _run_script(args):
    prog = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = prog.communicate()
    if out:
        print(out)
    if err:
        print(err)


def reduce_dimensions(data, random_state, target_dim=2):
    jl_limit = johnson_lindenstrauss_min_dim(n_samples=data.shape[0], eps=.3)
    pca_limit = 30

    if data.shape[1] > jl_limit and data.shape[1] > pca_limit:
        data = SparseRandomProjection(n_components=jl_limit, random_state=random_state).fit_transform(data)

    if data.shape[1] > pca_limit:
        data = PCA(n_components=pca_limit, random_state=random_state).fit_transform(data)

    return UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=random_state).fit_transform(data)


def reduce_dimensions_sparse(data, random_state, target_dim=2):
    svd_limit = 30

    if data.shape[1] > svd_limit:
        data = TruncatedSVD(n_components=svd_limit, random_state=random_state).fit_transform(data)
        data = Normalizer(copy=False).fit_transform(data)

    return UMAP(n_components=target_dim, random_state=random_state).fit_transform(data)
