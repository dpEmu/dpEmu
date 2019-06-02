import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters
import src.problemgenerator.utils as utils

data = np.array(["shambler", "shub-niggurath", "ogre", "difficulty: nightmare",
                 "quad damage", "health 100, health 99, health 0"])

x_node = array.Array(data.shape)
params = utils.load_ocr_error_params("example_ocr_error_params.json")

for i in range(100000):
    normalized_params = utils.normalize_ocr_error_params(params, p=.5)

for c in ["a", "q", "1", ":"]:
    print(c, params[c], normalized_params[c])

x_node.addfilter(filters.OCRError(normalized_params))
root_node = copy.Copy(x_node)
out = root_node.process(data)
print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
