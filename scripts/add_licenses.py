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

import os
import sys
from pathlib import Path

ignored_dirs = ["libs", "venv", "examples/speech_commands", "docs"]
license_file = open(sys.argv[1], 'r')
license_text = license_file.read()
license_file.close()

# remove newlines from the end
while license_text[-1] == '\n':
    license_text = license_text[:-1]

# make each line a block comment
license_text = "# " + "\n# ".join(license_text.split("\n")) + "\n\n"

# replace empty license lines i.e. "# " with just "#"
license_text = license_text.replace("# \n", "#\n")

# iterate over all descendant python files
pathlist = Path(str(os.getcwd())).glob('**/*.py')
for path in pathlist:
    path_in_str = str(path)

    # check that the path is not on the list of ignored directories
    valid = True
    for directory in ignored_dirs:
        path = str(os.getcwd()) + "/" + directory + "/"
        if path_in_str.startswith(path):
            valid = False
            break

    if valid:
        # add license to the source code
        source_file = open(path_in_str, 'r')
        source = source_file.read()
        if source.startswith(license_text[:-2]):  # ignore two last newlines
            continue
        source_file.close()

        source = license_text + source
        while source[-2:] == '\n\n':
            source = source[:-1]

        new_file = open(path_in_str, 'w')
        new_file.write(source)
        new_file.close()
        print("Added the license to", path_in_str)
