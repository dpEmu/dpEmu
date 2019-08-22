# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from dpemu.nodes import Array
from dpemu.filters.text import MissingArea
from dpemu import radius_generators

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

root_node = Array()
root_node.addfilter(MissingArea("p", "radius_gen", "value"))

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
