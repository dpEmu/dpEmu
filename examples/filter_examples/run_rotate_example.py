import sys
import matplotlib.pyplot as plt

from dpemu.nodes import Array
from dpemu.filters.image import Rotation


def main():
    angle = float(sys.argv[1])
    data = plt.imread("data/landscape.png")
    print(type(data))
    print(data.shape)
    root_node = Array()
    root_node.addfilter(Rotation("angle"))
    result = root_node.generate_error(data, {'angle': angle})

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
