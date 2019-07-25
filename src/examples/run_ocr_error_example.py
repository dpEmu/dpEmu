from time import time

import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.utils as utils

data = np.array(["shambler", "shub-niggurath", "ogre", "difficulty: nightmare",
                 "quad damage", "health 100, health 99, health 0"])

t0 = time()
for i in range(20000):
    root_node = array.Array(data.shape)
    params = utils.load_ocr_error_params("config/example_text_error_params.json")
    normalized_params = utils.normalize_ocr_error_params(params)

    root_node.addfilter(filters.OCRError("params", "p"))
    out = root_node.generate_error(data, {'params': normalized_params, 'p': .5})

print("Time:", time() - t0)

for c in ["a", "q", "1", ":"]:
    print(c, params[c], normalized_params[c])

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
