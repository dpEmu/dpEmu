import sys
import cv2
# from PIL import Image


import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    angle = float(sys.argv[1])
    data = cv2.imread("demo/landscape.png")
    print(type(data))
    print(data.shape)
    root_node = array.Array()
    root_node.addfilter(filters.Rotation("angle"))
    result = root_node.generate_error(data, {'angle': angle})
    # filtered_img = Image.fromarray(result)
    # filtered_img.show()
    cv2.imshow("Rotated", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
