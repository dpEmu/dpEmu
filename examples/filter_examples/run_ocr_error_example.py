import numpy as np
from dpemu.nodes import Array
from dpemu.filters.text import OCRError
from dpemu import pg_utils

data = np.array(["shambler", "shub-niggurath", "ogre", "difficulty: nightmare",
                 "quad damage", "health 100, health 99, health 0"])

root_node = Array()
params = pg_utils.load_ocr_error_params("data/example_text_error_params.json")
normalized_params = pg_utils.normalize_ocr_error_params(params)
root_node.addfilter(OCRError("params", "p"))
out = root_node.generate_error(data, {'params': normalized_params, 'p': .5})

for c in ["a", "q", "1", ":"]:
    print(c, params[c], normalized_params[c])

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
