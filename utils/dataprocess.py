import json


def read_json_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data
