import numpy as np

from PIL import Image

from dpemu.problemgenerator import array
from dpemu.problemgenerator import filters


def main():
    img = Image.open("demo/landscape.png")
    data = np.array(img)
    root_node = array.Array()
    root_node.addfilter(filters.JPEG_Compression('quality'))
    result = root_node.generate_error(data, {'quality': 5})
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
