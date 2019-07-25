import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    img_path = "demo/landscape.png"
    d = {"tar": 1, "rat": 0.55, "range": 255}

    # Use the vectorized version
    img2 = Image.open(img_path)
    data = np.array(img2)
    x_node = array.Array(data.shape)
    b2 = filters.BrightnessVectorized("tar", "rat", "range")
    x_node.addfilter(b2)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
