import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from src.problemgenerator.array import Array
from src.problemgenerator.filters import ClipWAV

dyn_range = float(sys.argv[1])

data_dir = "src/examples/speech_commands/example_data"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]
data = np.array([read(f)[1] for f in files])

root_node = Array()
root_node.addfilter(ClipWAV("dyn_range"))
params =  {"dyn_range": .3}

clipped = root_node.generate_error(data, params)
for i in range(len(data)):
    print(max(data[i]), max(clipped[i]))
