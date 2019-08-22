import numpy as np
from PIL import Image

from dpemu.nodes import Array
from dpemu.filters import Constant, Identity, Subtraction

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
root_node = Array()
# add filter which subtracts each pixel value from 255
root_node.addfilter(Subtraction("const", "identity"))
out = root_node.generate_error(data, {'c': 255, 'const': Constant("c"), 'identity': Identity()})

# show modified image
img_modified = Image.fromarray(out, "RGB")
img_modified.show()

print(out)
print("output shape:", out.shape, ", output dtype:", out.dtype)
