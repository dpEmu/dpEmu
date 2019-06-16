import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters
import src.problemgenerator.radius_generators as radius_generators

data = np.array(["Lorem ipsum dolor sit amet,\n" +
                 "consectetur adipisci elit,\n" +
                 "sed eiusmod tempor incidunt\n" +
                 "ut labore et dolore magna aliqua.\n" +
                 "Ut enim ad minim veniam,\n" +
                 "quis nostrud exercitation ullamco\n" +
                 "laboris nisi ut aliquid ex ea commodi consequat. \n" +
                 "Quis aute iure reprehenderit in voluptate\n" +
                 "velit esse cillum dolore eu fugiat nulla pariatur.\n" +
                 "Excepteur sint obcaecat cupiditat non proident,\n" +
                 "sunt in culpa qui officia\n" +
                 "deserunt mollit anim id est laborum.",
                 "Hello\n" +  # the next string starts here
                 "Hello\n" +
                 "Hello\n" +
                 "Hello\n" +
                 "Hello"])

y_node = array.Array(data.shape)
y_node.addfilter(filters.MissingArea(.02, radius_generators.GaussianRadiusGenerator(1, 1), " "))

root_node = copy.Copy(y_node)

out = root_node.process(data, np.random.RandomState(seed=42))
for index, elem in enumerate(out):
    print(elem, end="\n\n")

y_node = array.Array(data.shape)
# when a missing area is generated, its radius is 1 with probability of 0.6 and 2 with the probability of 0.4
y_node.addfilter(filters.MissingArea(.02, radius_generators.ProbabilityArrayRadiusGenerator([0, 0.6, 0.4]), " "))

root_node = copy.Copy(y_node)

out = root_node.process(data, np.random.RandomState(seed=42))
for index, elem in enumerate(out):
    print(elem, end="\n\n")
