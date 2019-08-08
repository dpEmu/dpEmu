import numpy as np

from dpemu import array
from dpemu import filters

data = np.array(["hello world",
                 "all your Bayes' theorems are belong to us"])

root_node = array.Array()
root_node.addfilter(filters.Uppercase("prob"))
out = root_node.generate_error(data, {'prob': .45})
print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
