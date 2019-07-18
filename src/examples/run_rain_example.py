import numpy as np

import matplotlib.pyplot as plt
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def img_to_pixel_data(img):
    pixels = img.load()
    data = np.zeros((img.size[1], img.size[0]), dtype=(int, 3))
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            data[y][x] = pixels[x, y]
    return data


def main():
    d = {"prob": 0.01}
    # img = Image.open("data/val2017/000000000776.jpg")
    # data = img_to_pixel_data(img)
    # x_node.addfilter(filters.Snow(0.01, 0.4, 1))

    data1 = plt.imread("data/val2017/000000000776.jpg")
    print("Original", data1)
    x_node1 = array.Array(data1.shape)
    x_node1.addfilter(filters.Rain("prob"))
    result1 = x_node1.generate_error(data1, d)

    plt.imshow(result1)
    plt.show()


if __name__ == "__main__":
    main()
