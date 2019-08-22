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

import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dpemu.nodes import Array
from dpemu.filters.image import Rain


def main():
    img_path = "data/landscape.png"
    img = Image.open(img_path)
    data = np.array(img)
    # data = plt.imread(img_path)

    root_node = Array()
    # root_node.addfilter(Snow("p", "flake_alpha", "storm_alpha"))
    root_node.addfilter(Rain("p", "r"))
    before = time.time()
    result = root_node.generate_error(data, {'p': .01, 'r': 255})
    end = time.time()

    print(f"{end - before} faster time")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
