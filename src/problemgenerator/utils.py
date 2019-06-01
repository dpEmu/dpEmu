import json


def load_ocr_error_frequencies(error_frequencies_config):
    return json.load(open(error_frequencies_config))


def create_normalized_probs(params):
    replacements = {}
    for key, weight_pairs in params.items():
        replacements[key] = (weight_pairs[0], normalize(weight_pairs[1]))

    return replacements


def normalize(weights):
    support = sum(weights)
    normalized_weights = []
    for weight in weights:
        normalized_weights.append(weight / support)

    return normalized_weights
