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

import cv2
import numpy as np

from dpemu.nodes import Array
from dpemu.filters import Addition, Constant, Division
from dpemu.filters.image import Rotation
from dpemu.filters.common import ModifyAsDataType


def main():
    data = cv2.imread("data/landscape.png")
    x_node = Array()

    # Some filters, e.g. Rotation, expect the data to have a specific data type, e.g. uint8.
    #
    # This example takes an image, sums the original image and 180 degrees rotated version,
    # and then takes the average of each pixel's value.
    #
    # cv2's rotation requires data to be uint8, but summing them needs datatype with larger
    # precision, and thus type conversions are required.

    const = Constant("c")
    rot1 = Rotation("deg1")
    mod1 = ModifyAsDataType("rotation_dtype", rot1)
    rot2 = Rotation("deg2")
    mod2 = ModifyAsDataType("rotation_dtype", rot2)
    add = Addition(mod1, mod2)
    avg = Division(add, const)
    x_node.addfilter(ModifyAsDataType("avg_dtype", avg))

    params = {}
    params['c'] = 2
    params['rotation_dtype'] = np.uint8
    params['avg_dtype'] = np.uint16
    params['deg1'] = 0
    params['deg2'] = 180

    result = x_node.generate_error(data, params)
    cv2.imshow("Rotated", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
