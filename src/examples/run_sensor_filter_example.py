import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy

y = np.arange(100.0, 200.0)
print("Original y:\n", y)

data = y
y_node = array.Array(y.shape)
print(f"input shape: {data.shape}")
root_node = copy.Copy(y_node)

"""
Every increase in time results in drift increasing by 0.1
y_node.addfilter(filters.SensorDrift(magnitude=0.1))
"""

"""
def strange(a):
    if a <= 170 and a >= 150:
        return 1729

    return a
"""
# y_node.addfilter(filters.StrangeBehaviour(strange))
y_node.addfilter(filters.Gap(prob_break=.3, prob_recover=.3, missing_value=np.nan))

output = root_node.process(data, np.random.RandomState(seed=42))
print("output:\n", output)
print(f"output dtype: {output.dtype}")
