import numpy as np

import src.problemgenerator.array as array
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

root_node = array.Array(data.shape)
root_node.addfilter(filters.MissingArea("p", "radius_gen", "value"))

params = {}
params['value'] = " "

params['p'] = .03
params['radius_gen'] = radius_generators.GaussianRadiusGenerator(1, 1)

out = root_node.generate_error(data, params)

for elem in out:
    print(elem, end="\n\n")

params['p'] = .04
params['radius_gen'] = radius_generators.ProbabilityArrayRadiusGenerator([0, 0.6, 0.4])

out = root_node.generate_error(data, params)
for elem in out:
    print(elem, end="\n\n")
