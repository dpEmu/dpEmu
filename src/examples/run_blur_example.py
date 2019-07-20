import numpy as np

from PIL import Image

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def main():
    img = Image.open("demo/landscape.png")
    data = np.array(img)
    x_node = array.Array(data.shape)
    x_node.addfilter(filters.Blur_Gaussian('std'))
    root_node = copy.Copy(x_node)
    root_node.set_error_params({'std': 10.0})
    result = root_node.process(data, np.random.RandomState(seed=42))
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()
