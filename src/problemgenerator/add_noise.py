import numpy as np
import problemgenerator.array as array
import problemgenerator.filter as filter
import problemgenerator.series as series


def add_noise_to_imgs(x_file, y_file, std):

    x = np.load(x_file)
    y = np.load(y_file)
    data = (x, y)

    x_node = array.Array(x[0].shape)
    x_node.addfilter(filter.GaussianNoise(0, std))
    y_node = array.Array(y[0].shape)
    root_node = series.TupleSeries([x_node, y_node])

    return root_node.process(data)
