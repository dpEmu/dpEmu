import json


def load_ocr_error_params(path_to_error_params):
    return json.load(open(path_to_error_params))


def normalize_ocr_error_params(params):
    return {k: (v[0], normalize_probs(v[1])) for k, v in params.items()}


def normalize_probs(probs):
    total = sum(probs)
    return [prob / total for prob in probs]
