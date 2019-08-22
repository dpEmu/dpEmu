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
from dpemu.filters import Filter


class MissingArea(Filter):
    """Emulate optical character recognition effect of stains in text.

    Introduce missing areas to text.

    Inherits Filter class.
    """
    # TODO: radius_generator is a struct, not a function. It should be a function for repeatability

    def __init__(self, probability_id, radius_generator_id, missing_value_id):
        """
        Args:
            probability_id (str): A key which maps to a probability of stain.
            radius_generator_id (str): A key which maps to a radius_generator.
            missing_value_id (str): A key which maps to a missing value to be used.
        """
        self.probability_id = probability_id
        self.radius_generator_id = radius_generator_id
        self.missing_value_id = missing_value_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        if self.probability == 0:
            return

        for index, _ in np.ndenumerate(node_data):
            # 1. Get indexes of newline characters. We will not touch those
            string = node_data[index]

            row_starts = [0]
            for i, c in enumerate(string):
                if c == '\n':
                    row_starts.append(i + 1)
            if not row_starts or row_starts[-1] != len(string):
                row_starts.append(len(string))
            height = len(row_starts) - 1

            widths = np.array([row_starts[i + 1] - row_starts[i] - 1 for i in range(height)])
            if len(widths) > 0:
                width = np.max(widths)
            else:
                width = 0

            # 2. Generate error
            errs = np.zeros(shape=(height + 1, width + 1))
            ind = -1
            while True:
                ind += random_state.geometric(self.probability)

                if ind >= width * height:
                    break
                y = ind // width
                x = ind - y * width
                r = self.radius_generator.generate(random_state)
                x0 = max(x - r, 0)
                x1 = min(x + r + 1, width)
                y0 = max(y - r, 0)
                y1 = min(y + r + 1, height)
                errs[y0, x0] += 1
                errs[y0, x1] -= 1
                errs[y1, x0] -= 1
                errs[y1, x1] += 1

            # 3. Perform prefix sums, create mask
            errs = np.cumsum(errs, axis=0)
            errs = np.cumsum(errs, axis=1)
            errs = (errs > 0)

            mask = np.zeros(len(string))
            for y in range(height):
                ind = row_starts[y]
                mask[ind:ind + widths[y]] = errs[y, 0:widths[y]]

            # 4. Apply error to string
            res_str = "".join([' ' if mask[i] else string[i] for i in range(len(mask))])
            node_data[index] = res_str


class OCRError(Filter):
    """Emulate optical character recognition (OCR) errors.

    User should provide a probability distribution in the form of a dict,
    specifying how probable a change of character is. Example weights for
    the distribution can be found in the data directory. These files are:

    example_text_error_params_realistic_ocr.json and
    example_text_error_params.json

    These weights can be loaded and the weights normalised into a probability
    distribution using functions from dpemu/pg_utils.py.

    Inherits Filter class.
    """

    def __init__(self, normalized_params_id, p_id):
        """
        Args:
            normalized_params_id (str): A key which maps to the probability distribution.
            p_id (str): A key which maps to a probability of the distribution being applied.
        """
        self.normalized_params_id = normalized_params_id
        self.p_id = p_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        for index, string_ in np.ndenumerate(node_data):
            node_data[index] = (self.generate_ocr_errors(string_, random_state))

    def generate_ocr_errors(self, string_, random_state):
        return "".join([self.replace_char(c, random_state) for c in string_])

    def replace_char(self, c, random_state):
        if c in self.normalized_params and random_state.random_sample() < self.p:
            chars, probs = self.normalized_params[c]
            return random_state.choice(chars, 1, p=probs)[0]

        return c


class Uppercase(Filter):
    """Randomly convert characters to uppercase.

    For each character in the string, convert the character
    to uppercase with the provided probability.

    Inherits Filter class.
    """

    def __init__(self, probability_id):
        """
        Args:
            probability_id (str): A key which maps to the probability of uppercase change.
        """
        self.prob_id = probability_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):

        def stochastic_upper(char, probability):
            if random_state.rand() <= probability:
                return char.upper()
            return char

        for index, element in np.ndenumerate(node_data):
            original_string = element
            modified_string = "".join(
                [stochastic_upper(c, self.prob) for c in original_string])
            node_data[index] = modified_string
