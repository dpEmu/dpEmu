import sys
import matplotlib.pyplot as plt

from dpemu.nodes import Array, Series
from dpemu.filters.image import Rotation
from dpemu.dataset_utils import load_mnist


def main():
    """An example that rotates MNIST digits and displays one.
    Usage: python run_rotate_MNIST_example <angle>
    where <angle> is the angle of rotation
    (e.g. 90 to rotate by pi / 2)
    """
    x, _, _, _ = load_mnist()
    xs = x[:20]                 # small subset of x
    angle = float(sys.argv[1])
    print(xs.shape)
    img_node = Array(reshape=(28, 28))
    root_node = Series(img_node)
    img_node.addfilter(Rotation("angle"))
    result = root_node.generate_error(xs, {'angle': angle})

    plt.matshow(result[0].reshape((28, 28)))
    plt.show()


if __name__ == "__main__":
    main()
