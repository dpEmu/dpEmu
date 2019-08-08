import sys
import cv2
import matplotlib.pyplot as plt


from dpemu import array
from dpemu import filters


def main():
    angle = float(sys.argv[1])
    data = cv2.imread("demo/landscape.png")
    print(type(data))
    print(data.shape)
    root_node = array.Array()
    root_node.addfilter(filters.Rotation("angle"))
    result = root_node.generate_error(data, {'angle': angle})
    # filtered_img = Image.fromarray(result)
    # filtered_img.show()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
