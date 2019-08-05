import numpy as np
from PIL import Image
from dpemu import array
from dpemu import filters


def main():
    img = Image.open("../demo/landscape.png")
    data = np.array(img)
    root_node = array.Array()
    root_node.addfilter(filters.Blur_Gaussian('std'))
    result = root_node.generate_error(data, {'std': 10.0})
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
