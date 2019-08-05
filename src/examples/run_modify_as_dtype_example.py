import cv2

import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    data = cv2.imread("demo/landscape.png")
    x_node = array.Array()

    # Some filters, e.g. Rotation, expect the data to have a specific data type, e.g. uint8.
    #
    # This example takes an image, sums the original image and 180 degrees rotated version,
    # and then takes the average of each pixel's value.
    #
    # cv2's rotation requires data to be uint8, but summing them needs datatype with larger
    # precision, and thus type conversions are required.

    const = filters.Constant("c")
    rot1 = filters.Rotation("deg1")
    mod1 = filters.ModifyAsDataType("rotation_dtype", "rot1")
    rot2 = filters.Rotation("deg2")
    mod2 = filters.ModifyAsDataType("rotation_dtype", "rot2")
    add = filters.Addition("mod1", "mod2")
    avg = filters.Division("add", "const")
    x_node.addfilter(filters.ModifyAsDataType("avg_dtype", "avg"))

    params = {}
    params['c'] = 2
    params['const'] = const
    params['add'] = add
    params['avg'] = avg
    params['rotation_dtype'] = np.uint8
    params['avg_dtype'] = np.uint16
    params['deg1'] = 0
    params['deg2'] = 180
    params['rot1'] = rot1
    params['rot2'] = rot2
    params['mod1'] = mod1
    params['mod2'] = mod2

    result = x_node.generate_error(data, params)
    cv2.imshow("Rotated", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
