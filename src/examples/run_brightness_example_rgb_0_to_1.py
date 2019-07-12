import time
import numpy as np
import matplotlib.pyplot as plt

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def main():
    d = {"tar": 1, "rat": 0.5, "range": 1}
    img_path = "demo/landscape.png"

    # Use the looped version
    data1 = plt.imread(img_path)
    x_node1 = array.Array(data1.shape)
    s1 = filters.Brightness("tar", "rat", "range")
    s1.set_params(d)
    x_node1.addfilter(s1)
    root_node1 = copy.Copy(x_node1)
    start1 = time.time()
    result1 = root_node1.process(data1, np.random.RandomState(seed=42))
    end1 = time.time()
    print(f"Time traditional: {end1-start1}")

    # Use the vectorized version
    data2 = plt.imread(img_path)
    x_node2 = array.Array(data2.shape)
    s2 = filters.BrightnessVectorized("tar", "rat", "range")
    s2.set_params(d)
    x_node2.addfilter(s2)
    root_node2 = copy.Copy(x_node2)
    start2 = time.time()
    result2 = root_node2.process(data2, np.random.RandomState(seed=42))
    end2 = time.time()
    print(f"Time vectorized: {end2-start2}")

    print()
    same = np.allclose(result1, result2)
    print(f"Arrays are near equal: {same}")
    if not same:
        print(result2-result1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(result1)
    ax[0].set_title("Traditional")
    ax[1].imshow(result2)
    ax[1].set_title("Vectorized")
    plt.show()


if __name__ == "__main__":
    main()
