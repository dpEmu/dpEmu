from dpemu import array
from dpemu import filters
from dpemu import series

# from src.plotting.utils import visualize_error_generator
from dpemu import plotting_utils

x_node = array.Array()
y_node = array.Array()
root_node = series.TupleSeries([x_node, y_node])

params = {"c": 0.5, "b": 5, "a": 1}
params['gauss_a'] = filters.GaussianNoise("a", "b")
params['gauss_b'] = filters.GaussianNoise("b", "a")

x_node.addfilter(filters.Addition('gauss_a', 'gauss_b'))
y_node.addfilter(filters.Missing(probability_id="c"))

plotting_utils.visualize_error_generator(root_node.get_parametrized_tree(params))
