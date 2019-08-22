# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
