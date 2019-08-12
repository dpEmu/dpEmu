import sys
import cv2
import matplotlib.pyplot as plt

from dpemu.nodes import Array
from dpemu.filters.image import Rotation


def main():
    angle = float(sys.argv[1])
    data = cv2.imread("demo/landscape.png")
    print(type(data))
    print(data.shape)
    root_node = Array()
    root_node.addfilter(Rotation("angle"))
    result = root_node.generate_error(data, {'angle': angle})
    # filtered_img = Image.fromarray(result)
    # filtered_img.show()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
