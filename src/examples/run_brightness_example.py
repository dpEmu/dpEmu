import numpy as np

# from PIL import Image
import matplotlib.pyplot as plt

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy


def main():
    # img = Image.open("demo/landscape.png")
    img = plt.imread("demo/landscape.png")
    print(img)
    data = np.array(img)
    x_node = array.Array(data.shape)
    d = {"tar": 1, "rat": 0.5}
    b = filters.Brightness("tar", "rat")
    b.set_params(d)
    x_node.addfilter(b)
    root_node = copy.Copy(x_node)
    result = root_node.process(data, np.random.RandomState(seed=42))
    plt.imshow(result)
    plt.show()
    # filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    # filtered_img.show()


if __name__ == "__main__":
    main()
