import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy
import src.problemgenerator.utils as utils

data = np.array(["shambler", "shub-niggurath", "ogre", "difficulty: nightmare",
                 "quad damage", "health 100, health 99, health 0"])

x_node = array.Array(data.shape)
error_params = utils.load_ocr_error_frequencies(
    "example_ocr_error_weights.json")
replacements = utils.create_normalized_probs(error_params)

x_node.addfilter(filters.OCRerror(0.2, replacements))
root_node = copy.Copy(x_node)
out = root_node.process(data)
print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
