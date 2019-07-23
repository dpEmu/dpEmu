import time
import numpy as np
import matplotlib.pyplot as plt

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    d = {"tar": 1, "rat": 1.1, "range": 1}
    img_path = "demo/landscape.png"

    # Use the looped version
    data1 = plt.imread(img_path)
    x_node1 = array.Array(data1.shape)
    s1 = filters.Saturation("tar", "rat", "range")
    x_node1.addfilter(s1)
    start1 = time.time()
    result1 = x_node1.generate_error(data1, d)
    end1 = time.time()
    print(f"Time traditional: {end1-start1}")

    # Use the vectorized version
    data2 = plt.imread(img_path)
    x_node2 = array.Array(data2.shape)
    s2 = filters.SaturationVectorized("tar", "rat", "range")
    x_node2.addfilter(s2)
    start2 = time.time()
    result2 = x_node2.generate_error(data2, d)
    end2 = time.time()
    print(f"Time vectorized: {end2-start2}")

    print()
    same = np.allclose(result1, result2)
    print(f"Arrays are near equal: {same}")
    if not same:
        print(result2-result1)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(result1)
    ax[0].set_title("Traditional")
    ax[1].imshow(result2)
    ax[1].set_title("Vectorized")
    plt.show()


if __name__ == "__main__":
    main()
