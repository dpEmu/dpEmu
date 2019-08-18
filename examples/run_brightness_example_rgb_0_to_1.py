import time
import matplotlib.pyplot as plt

from dpemu.nodes import Array
from dpemu.filters.image import Brightness


def main():
    d = {"tar": 1, "rat": 0.5, "range": 1}
    img_path = "demo/landscape.png"

    data = plt.imread(img_path)
    x_node = Array()
    b = Brightness("tar", "rat", "range")
    x_node.addfilter(b)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
