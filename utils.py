import json


def minmaxscale(x):
    return (x - min(x)) / (max(x) - min(x))


def inverse_minmaxscale(x, min_x, max_x):
    return x * (max_x - min_x) + min_x


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
