import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters

data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

params = {"a": [["e"], [1.0]]}
ocr = filters.OCRError(params, p=1.0)

x_node = array.Array(data.shape)
x_node.addfilter(filters.ApplyWithProbability(ocr, 0.5))
series_node = series.Series(x_node)
root_node = copy.Copy(series_node)
out = root_node.process(data, np.random.RandomState(seed=42))

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
