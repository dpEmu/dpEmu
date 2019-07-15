import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy

from src.plotting.utils import visualize_error_generator

x_node = array.Array((20,))
y_node = array.Array((10, 2))
series_node = series.TupleSeries([x_node, y_node])
root_node = copy.Copy(series_node)

x_node.addfilter(filters.Addition(filters.GaussianNoise("a", "b"), filters.GaussianNoise("b", "a")))
y_node.addfilter(filters.Missing(probability_id="c"))

root_node.generate_error((np.array([1.0]), np.array([1.0])), {"c": 0.5, "b": 5, "a": 1})

visualize_error_generator(root_node)
