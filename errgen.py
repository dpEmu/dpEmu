import sys
import numpy as np
import problemgenerator.tensor as tensor
import problemgenerator.filter as filter
import problemgenerator.series as series

# datafile = sys.argv[1]
# data = np.genfromtxt(datafile, delimiter=',')
data = np.random.randn(100, 10)  # 100 rows, 10 cols

t = tensor.Tensor(data.shape[1])
t.addfilter(filter.Missing(probability=.3))
s = series.Series(t)
out = s.process(data)
print("input data has shape", data.shape)
print("output data has shape", out.shape)
print("relative frequency of NaNs:", np.isnan(out).sum() / out.size)