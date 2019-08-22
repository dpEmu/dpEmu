import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dpemu.nodes import Array
from dpemu.filters.image import Brightness


def main():
    img_path = "data/landscape.png"
    d = {"tar": 1, "rat": 0.55, "range": 255}

    img2 = Image.open(img_path)
    data = np.array(img2)
    x_node = Array()
    b2 = Brightness("tar", "rat", "range")
    x_node.addfilter(b2)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
