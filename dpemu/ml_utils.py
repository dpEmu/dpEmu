# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from os.path import isfile
from shlex import split
from subprocess import Popen, PIPE, STDOUT
from sys import stdout

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP

from dpemu.utils import get_project_root


def run_ml_module_using_cli(cline, show_stdout=True):
    """Runs an external ML model using its CLI.

    Args:
        cline: Command line used to call the external ML model.
        show_stdout: True to print the stdout of the external ML model.

    Returns:
        A string containing the stdout of the external ML model.
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
    """
    Reduces the dimensionality of the data using UMAP for lower dimensions, PCA for higher dimensions and possibly
    even random projections if the number of dimension is over the limit given by the Johnson–Lindenstrauss lemma. Works
    for NumPy arrays.

    Args:
        data: The input data.
        random_state: Random state to generate reproducible results.
        target_dim: The targeted dimension.

    Returns:
        Lower dimension representation of the data.
    """
    jl_limit = johnson_lindenstrauss_min_dim(n_samples=data.shape[0], eps=.3)
    pca_limit = 30

    if data.shape[1] > jl_limit and data.shape[1] > pca_limit:
        data = SparseRandomProjection(n_components=jl_limit, random_state=random_state).fit_transform(data)

    if data.shape[1] > pca_limit:
        data = PCA(n_components=pca_limit, random_state=random_state).fit_transform(data)

    return UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=random_state).fit_transform(data)


def reduce_dimensions_sparse(data, random_state, target_dim=2):
    """
    Reduces the dimensionality of the data using UMAP for lower dimensions and TruncatedSVD for higher dimensions. Works
    for SciPy sparse matrices.

    Args:
        data: The input data.
        random_state: Random state to generate reproducible results.
        target_dim: The targeted dimension.

    Returns:
        Lower dimension representation of the data.
    """
    svd_limit = 30

    if data.shape[1] > svd_limit:
        data = TruncatedSVD(n_components=svd_limit, random_state=random_state).fit_transform(data)

    return UMAP(n_components=target_dim, random_state=random_state).fit_transform(data)


def load_yolov3():
    """Loads the custom weights and cfg for the YOLOv3 model.

    Returns:
        Paths to YOLOv3 weights and cfg file.
    """
    path_to_yolov3_weights = f"{get_project_root()}/tmp/yolov3-spp_best.weights"
    if not isfile(path_to_yolov3_weights):
        Popen(["./scripts/get_yolov3.sh"], cwd=get_project_root()).wait()
    return path_to_yolov3_weights, f"{get_project_root()}/tmp/yolov3-spp.cfg"
