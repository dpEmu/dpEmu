import numpy as np

from PIL import Image

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
    img = Image.open("data/val2017/000000000776.jpg")
    data = img_to_pixel_data(img)
    root_node = array.Array(data.shape)
    root_node.addfilter(filters.Snow("p", "flake_alpha", "storm_alpha"))
    # root_node.addfilter(filters.Rain("p"))
    result = root_node.generate_error(data, {'p': .01, 'flake_alpha': .4, 'storm_alpha': 1.})
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
