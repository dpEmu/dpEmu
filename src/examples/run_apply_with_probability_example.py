import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.series as series
import src.problemgenerator.filters as filters

data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

params = {"a": [["e"], [1.0]]}
ocr = filters.OCRError("ocr_params", "ocr_p")

x_node = array.Array(data.shape)
x_node.addfilter(filters.ApplyWithProbability('ocr', 'p'))
root_node = series.Series(x_node)

out = root_node.generate_error(data, {'ocr_params': params, 'ocr_p': 1.0, 'ocr': ocr, 'p': 0.5})

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
