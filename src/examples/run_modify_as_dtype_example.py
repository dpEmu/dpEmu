import cv2

import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def rotate(deg):
    return filters.ModifyAsDataType(np.uint8, filters.Rotation(deg))


def main():
    data = cv2.imread("demo/landscape.png")
    x_node = array.Array(data.shape)

    # Some filters, e.g. Rotation, expect the data to have a specific data type, e.g. uint8.
    #
    # This example takes an image, sums the original image and 180 degrees rotated version,
    # and then takes the average of each pixel's value.
    #
    # cv2's rotation requires data to be uint8, but summing them needs datatype with larger
    # precision, and thus type conversions are required.
    const = filters.Constant(2)
    avg = filters.Division(filters.Addition(rotate(0.0), rotate(180.0)), const)
    x_node.addfilter(filters.ModifyAsDataType(np.int16, avg))

    root_node = copy.Copy(x_node)
    result = root_node.process(data, np.random.RandomState(seed=42))
    cv2.imshow("Rotated", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
