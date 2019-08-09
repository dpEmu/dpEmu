import numpy as np

from PIL import Image

from dpemu.problemgenerator import array
from dpemu.problemgenerator import filters
from dpemu import radius_generators


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
    root_node = array.Array()
    root_node.addfilter(filters.StainArea("p", "radius_gen", "alpha"))
    params = {'p': .00002, "radius_gen": radius_generators.GaussianRadiusGenerator(50, 20), 'alpha': .9}
    result = root_node.generate_error(data, params)
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
