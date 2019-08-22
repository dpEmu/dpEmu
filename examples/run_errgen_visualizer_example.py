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
