import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dpemu.nodes import Array
from dpemu.filters.image import Saturation


def main():
    img_path = "demo/landscape.png"
    d = {"tar": 0, "rat": 0.8, "range": 255}

    img = Image.open(img_path)
    data = np.array(img)
    x_node = Array()
    s = Saturation("tar", "rat", "range")
    x_node.addfilter(s)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
