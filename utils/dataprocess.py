import json


def load_data(data_path):
    res = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            res.append(json.loads(line))
    return res
