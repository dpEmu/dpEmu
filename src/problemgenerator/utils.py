import json


def load_ocr_error_frequencies(error_frequencies_config):
    return json.load(open(error_frequencies_config))


def create_normalized_probs(params, p):
    replacements = {}
    for key, weight_pairs in params.items():
        chars, probs = weight_pairs
        chars = chars[1:]
        chars.append(key)
        probs = normalize(probs, p)
        replacements[key] = (chars, probs)

    return replacements


def normalize(weights, p):
    support = sum(weights)
    normalized_weights = []
    for weight in weights[1:]:
        normalized_weights.append(p * weight / support)

    normalized_weights.append(1 - sum(normalized_weights))
    return normalized_weights
