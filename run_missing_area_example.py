import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.copy as copy
import src.problemgenerator.series as series

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
                 "deserunt mollit anim id est laborum."])

x_node = array.Array(data.shape)
x_node.addfilter(filters.MissingArea(0.03, 1, 1, " "))

root_node = copy.Copy(x_node)

out = root_node.process(data)
for index, elem in enumerate(out):
    print("Line" + str(index + 1) + ":")
    print(elem, end="\n\n")
print("output shape:", out.shape, ", output dtype:", out.dtype)


