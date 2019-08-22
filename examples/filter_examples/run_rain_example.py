import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dpemu.nodes import Array
from dpemu.filters.image import Rain


def main():
    img_path = "data/landscape.png"
    img = Image.open(img_path)
    data = np.array(img)
    # data = plt.imread(img_path)

    root_node = Array()
    # root_node.addfilter(Snow("p", "flake_alpha", "storm_alpha"))
    root_node.addfilter(Rain("p", "r"))
    before = time.time()
    result = root_node.generate_error(data, {'p': .01, 'r': 255})
    end = time.time()

    print(f"{end - before} faster time")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
