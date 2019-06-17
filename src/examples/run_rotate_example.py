import sys
import cv2
import numpy as np
# from PIL import Image


import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def main():
    angle = float(sys.argv[1])
    data = cv2.imread("demo/landscape.png")
    print(type(data))
    print(data.shape)
    x_node = array.Array(data.shape)
    x_node.addfilter(filters.Rotation(angle))
    root_node = copy.Copy(x_node)
    result = root_node.process(data, np.random.RandomState(seed=42))
    # filtered_img = Image.fromarray(result)
    # filtered_img.show()
    cv2.imshow("Rotated", result)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
