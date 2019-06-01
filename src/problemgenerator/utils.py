import json
from copy import deepcopy


def load_ocr_error_frequencies(error_frequencies_config):
    return json.load(open(error_frequencies_config))


def create_normalized_probs(params, p):
    replacements = {}
    for key, weight_pairs in params.items():
        chars, probs = deepcopy(weight_pairs)
        key_idx = chars.index(key)
        chars.pop(key_idx)
        probs = normalize(probs, key_idx, p)
        chars.append(key)
        replacements[key] = (chars, probs)

    return replacements


def normalize(weights, key_idx, p):
    support = sum(weights)
    weights.pop(key_idx)
    normalized_weights = []
    for weight in weights:
        normalized_weights.append(p * weight / support)

    normalized_weights.append(1 - sum(normalized_weights))
    return normalized_weights
