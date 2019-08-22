from dpemu import plotting_utils
from dpemu.filters import Addition
from dpemu.filters.common import GaussianNoise, Missing
from dpemu.nodes import Array, TupleSeries


x_node = Array()
y_node = Array()
root_node = TupleSeries([x_node, y_node])

params = {}
params['probability'] = 0.5
params['missing_value'] = 0
params['mean_a'] = 1
params['mean_b'] = 2
params['std_a'] = 4
params['std_b'] = 3
gaussian_a = GaussianNoise("mean_a", "std_a")
gaussian_b = GaussianNoise("mean_b", "std_b")

x_node.addfilter(Addition(gaussian_a, gaussian_b))
y_node.addfilter(Missing("probability", "missing_value"))

plotting_utils.visualize_error_generator(root_node.get_parametrized_tree(params))
