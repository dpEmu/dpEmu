from time import time

import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters
import src.problemgenerator.utils as utils

data = np.array(["shambler", "shub-niggurath", "ogre", "difficulty: nightmare",
                 "quad damage", "health 100, health 99, health 0"])

t0 = time()
for i in range(20000):
    x_node = array.Array(data.shape)
    params = utils.load_ocr_error_params(
        "config/example_ocr_error_params_1.json")

    normalized_params = utils.normalize_ocr_error_params(params)

    x_node.addfilter(filters.OCRError(normalized_params, p=.5))
    root_node = copy.Copy(x_node)
    out = root_node.process(data)

print("Time:", time() - t0)

for c in ["a", "q", "1", ":"]:
    print(c, params[c], normalized_params[c])

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
