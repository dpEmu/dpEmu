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
