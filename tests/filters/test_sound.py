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
from dpemu.filters.sound import ClipWAV


def test_clip_wav():
    rng = np.random.RandomState(1729)
    dyn_ranges = [0.0, 0.22, 0.5, 0.73, 1.0]

    for dyn_range in dyn_ranges:
        data = rng.rand(100) * 100
        min_, max_ = np.amin(data), np.amax(data)
        new_half_range = 0.5 * (max_ - min_) * dyn_range
        mid = (max_ + min_) / 2
        new_data = np.clip(data, mid - new_half_range, mid + new_half_range)

        clip_wav = ClipWAV("dyn_range")
        clip_wav.set_params({"dyn_range": dyn_range})
        clip_wav.apply(data, rng, named_dims={})

        assert np.allclose(new_data, data)
