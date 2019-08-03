import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series

from src.plotting.utils import visualize_error_generator

x_node = array.Array()
y_node = array.Array()
root_node = series.TupleSeries([x_node, y_node])

params = {"c": 0.5, "b": 5, "a": 1}
params['gauss_a'] = filters.GaussianNoise("a", "b")
params['gauss_b'] = filters.GaussianNoise("b", "a")

x_node.addfilter(filters.Addition('gauss_a', 'gauss_b'))
y_node.addfilter(filters.Missing(probability_id="c"))

visualize_error_generator(root_node.get_parametrized_tree(params))
