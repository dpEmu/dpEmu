import numpy as np
from PIL import Image
from pathlib import Path
from dpemu import array
from dpemu import filters


def main():
    img = Image.open(Path(__file__).resolve().parents[1] / "demo/landscape.png")
    data = np.array(img)
    root_node = array.Array()
    # root_node.addfilter(filters.Blur_Gaussian('std'))
    # result = root_node.generate_error(data, {'std': 10.0})
    root_node.addfilter(filters.Blur('repeats', 'radius'))
    result = root_node.generate_error(data, {'repeats': 1, 'radius': 20})
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
