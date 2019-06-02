import json


def load_ocr_error_params(filename):
    return json.load(open(filename))


def normalize_ocr_error_params(params, p):
    normalized_params = {}
    for k, v in params.items():
        chars, probs = v
        key_idx = chars.index(k)
        normalized_probs = normalize_probs(probs, key_idx, p)
        normalized_params[k] = (chars, normalized_probs)

    return normalized_params


def normalize_probs(probs, key_idx, p):
    total = sum(probs)
    normalized_probs = [p * probs[i] / total for i in range(len(probs)) if i is not key_idx]
    normalized_probs.insert(key_idx, 1 - sum(normalized_probs))
    return normalized_probs
