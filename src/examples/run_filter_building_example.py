import numpy as np
from PIL import Image

import src.problemgenerator.array as array
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters

# generate image with bitwise operations
data = []
for y in range(0, 512):
    data.append([])
    for x in range(0, 512):
        data[y].append((x ^ y, x & y, 0))
data = np.array(data, dtype=np.uint8)

# show original image
img_original = Image.fromarray(data, "RGB")
img_original.show()

# generate error
x_node = array.Array(data.shape)
# add filter which subtracts each pixel value from 255
x_node.addfilter(filters.Subtraction(filters.Constant(255), filters.Identity()))
root_node = copy.Copy(x_node)
out = root_node.process(data, np.random.RandomState(seed=42))

# show modified image
img_modified = Image.fromarray(out, "RGB")
img_modified.show()

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
