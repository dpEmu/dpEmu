import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy


def add_noise_to_imgs(x_file, y_file, std):

    x = np.load(x_file)
    y = np.load(y_file)

    x_node = array.Array(x[0].shape)
    x_node.addfilter(filters.GaussianNoise(0, std))
    y_node = array.Array(y[0].shape)
    series_node = series.TupleSeries([x_node, y_node])
    root_node = copy.Copy(series_node)
    out_x, out_y = root_node.process((x, y))

    return [(out_x, out_y), (x, y)]
