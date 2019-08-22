import time

import matplotlib.pyplot as plt
from dpemu.nodes import Array
from dpemu.filters.image import Resolution


def main():
    d = {"resolution": 10}
    data = plt.imread("data/landscape.png")
    x_node = Array()
    r = Resolution("resolution")
    x_node.addfilter(r)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
