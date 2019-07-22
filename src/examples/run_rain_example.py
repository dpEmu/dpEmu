import time
import matplotlib.pyplot as plt

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters


def main():
    img_path = "demo/landscape.png"
    data = plt.imread(img_path)

    times = []
    results = []
    for i in range(2):
        root_node = array.Array(data.shape)
        # root_node.addfilter(filters.Snow("p", "flake_alpha", "storm_alpha"))
        if i == 0:
            root_node.addfilter(filters.Rain("p"))
        elif i == 1:
            root_node.addfilter(filters.FastRain("p", "r"))

        before = time.time()
        result = root_node.generate_error(data, {'p': .01, 'r': 1})
        times.append(time.time() - before)
        results.append(result)

    print(f"{times[0]} original time")
    print(f"{times[1]} faster time")

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(results[0])
    ax[0].set_title("Original")
    ax[1].imshow(results[1])
    ax[1].set_title("Faster")
    plt.show()


if __name__ == "__main__":
    main()
