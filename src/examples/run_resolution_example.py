import numpy as np
import time

# from PIL import Image
import matplotlib.pyplot as plt
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    d = {"resolution": 10}
    # data = plt.imread("demo/landscape.png")
    # print(data)

    # Traditional
    # img1 = Image.open("demo/landscape.png")
    # data1 = np.array(img1)
    data1 = plt.imread("demo/landscape.png")
    x_node1 = array.Array(data1.shape)
    r1 = filters.Resolution("resolution")
    x_node1.addfilter(r1)
    start1 = time.time()
    result1 = x_node1.generate_error(data1, d)
    end1 = time.time()
    print(f"Time traditional: {end1-start1}")

    # Vectorized
    # img2 = Image.open("demo/landscape.png")
    # data2 = np.array(img2)
    data2 = plt.imread("demo/landscape.png")
    x_node2 = array.Array(data2.shape)
    r2 = filters.ResolutionVectorized("resolution")
    x_node2.addfilter(r2)
    start2 = time.time()
    result2 = x_node2.generate_error(data2, d)
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
