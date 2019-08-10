import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dpemu.problemgenerator import array
from dpemu.problemgenerator import filters


def main():
    img_path = "demo/landscape.png"
    img = Image.open(img_path)
    data = np.array(img)
    # data = plt.imread(img_path)

    root_node = array.Array()
    # root_node.addfilter(filters.Snow("p", "flake_alpha", "storm_alpha"))
    root_node.addfilter(filters.FastRain("p", "r"))
    before = time.time()
    result = root_node.generate_error(data, {'p': .01, 'r': 255})
    end = time.time()

    print(f"{end - before} faster time")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()