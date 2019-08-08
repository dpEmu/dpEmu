import numpy as np
from dpemu import array
from dpemu import series
from dpemu import filters

data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

params = {"a": [["e"], [1.0]]}
ocr = filters.OCRError("ocr_params", "ocr_p")

x_node = array.Array()
x_node.addfilter(filters.ApplyWithProbability('ocr', 'p'))
root_node = series.Series(x_node)

out = root_node.generate_error(data, {'ocr_params': params, 'ocr_p': 1.0, 'ocr': ocr, 'p': 0.5})

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
