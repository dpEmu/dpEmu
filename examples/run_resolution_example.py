import time

# from PIL import Image
import matplotlib.pyplot as plt
from dpemu.problemgenerator import array
from dpemu.problemgenerator import filters


def main():
    d = {"resolution": 10}
    # data = plt.imread("demo/landscape.png")

    # Vectorized
    # img2 = Image.open("demo/landscape.png")
    # data2 = np.array(img2)
    data = plt.imread("demo/landscape.png")
    x_node = array.Array()
    r = filters.ResolutionVectorized("resolution")
    x_node.addfilter(r)
    start = time.time()
    result = x_node.generate_error(data, d)
    end = time.time()
    print(f"Time vectorized: {end-start}")

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
