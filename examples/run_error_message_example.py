import numpy as np
from dpemu.filters.common import GaussianNoise
from dpemu.nodes import Array

"""This example raises an exception (on purpose).
The parameter identifier "mean" is misspelled in the
params dictionary. This should result in an exception
with a helpful error message.
"""

xs = np.random.rand(100, 200)
array_node = Array()
array_node.addfilter(GaussianNoise("mean", "std"))
params = {"meany": 0.0, "std": 20.0}
errorified = array_node.generate_error(xs, params)
