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

import numpy as np

from dpemu.nodes import Array
from dpemu.filters.time_series import Gap

y = np.arange(100.0, 200.0)
print("Original y:\n", y)

data = y
root_node = Array()
print(f"input shape: {data.shape}")

"""
Every increase in time results in drift increasing by 0.1
root_node.addfilter(SensorDrift(magnitude=0.1))
"""

"""
def strange(a):
    if a <= 170 and a >= 150:
        return 1729

    return a
"""
# y_node.addfilter(StrangeBehaviour(strange))
root_node.addfilter(Gap("prob_break", "prob_recover", "missing_value"))

output = root_node.generate_error(data, {'prob_break': .3, 'prob_recover': .3, 'missing_value': np.nan})
print("output:\n", output)
print(f"output dtype: {output.dtype}")
