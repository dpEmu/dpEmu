from dpemu.nodes import Array, TupleSeries
from dpemu.problemgenerator.filters import GaussianNoise, Addition, Missing

# from src.plotting.utils import visualize_error_generator
from dpemu import plotting_utils

x_node = Array()
y_node = Array()
root_node = TupleSeries([x_node, y_node])

params = {"c": 0.5, "b": 5, "a": 1}
params['gauss_a'] = GaussianNoise("a", "b")
params['gauss_b'] = GaussianNoise("b", "a")

x_node.addfilter(Addition('gauss_a', 'gauss_b'))
y_node.addfilter(Missing(probability_id="c"))

plotting_utils.visualize_error_generator(root_node.get_parametrized_tree(params))
