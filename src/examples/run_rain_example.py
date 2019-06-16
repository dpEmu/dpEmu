import numpy as np

from PIL import Image

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def img_to_pixel_data(img):
    pixels = img.load()
    data = np.zeros((img.size[1], img.size[0]), dtype=(int, 3))
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            data[y][x] = pixels[x, y]
    return data


def main():
    img = Image.open("demo/landscape.png")
    data = img_to_pixel_data(img)
    x_node = array.Array(data.shape)
    #x_node.addfilter(filters.Snow(0.01, 0.4, 1))
    x_node.addfilter(filters.Rain(0.01))
    root_node = copy.Copy(x_node)
    result = root_node.process(data, np.random.RandomState(seed=42))
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
