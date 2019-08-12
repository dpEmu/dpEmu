import numpy as np
from dpemu.filters import Filter


class ClipWAV(Filter):
    def __init__(self, dyn_range_id):
        super().__init__()
        self.dyn_range_id = dyn_range_id

    def set_params(self, params_dict):
        self.dyn_range = params_dict[self.dyn_range_id]

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
