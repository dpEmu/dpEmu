import numpy as np

from dpemu.nodes import Array
from dpemu.filters.text import Uppercase

data = np.array(["hello world",
                 "all your Bayes' theorems are belong to us"])

root_node = Array()
root_node.addfilter(Uppercase("prob"))
out = root_node.generate_error(data, {'prob': .45})
print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
