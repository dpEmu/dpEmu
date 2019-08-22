import numpy as np
from dpemu.nodes import Array, Series
from dpemu.filters.text import OCRError
from dpemu.filters.common import ApplyWithProbability

data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

params = {"a": [["e"], [1.0]]}
ocr = OCRError("ocr_params", "ocr_p")

x_node = Array()
x_node.addfilter(ApplyWithProbability('ocr', 'p'))
root_node = Series(x_node)

out = root_node.generate_error(data, {'ocr_params': params, 'ocr_p': 1.0, 'ocr': ocr, 'p': 0.5})

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
