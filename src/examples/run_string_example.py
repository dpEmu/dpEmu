import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy

data = np.array(["hello world",
                 "all your Bayes' theorems are belong to us"])

x_node = array.Array(data.shape)
x_node.addfilter(filters.Uppercase(.45))
root_node = copy.Copy(x_node)
out = root_node.process(data, np.random.RandomState(seed=42))
print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
