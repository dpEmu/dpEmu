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
