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


class ClipWAV(Filter):
    def __init__(self, dyn_range_id):
        super().__init__()
        self.dyn_range_id = dyn_range_id

    def apply(self, node_data, random_state, named_dims):

        def clip_audio(data, dyn_range):
            min_, max_ = min(data), max(data)
            half_range = .5 * max_ - .5 * min_
            middle = (min_ + max_) / 2
            new_half_range = half_range * dyn_range
            upper_limit = middle + new_half_range
            lower_limit = middle - new_half_range
            return np.clip(data, lower_limit, upper_limit).astype(data.dtype)

        node_data[:] = clip_audio(node_data, self.dyn_range)
