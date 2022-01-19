import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save2json(data, filename):
    with open(filename, "w", encoding='UTF8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def loadFromJson(filename) -> dict:
    # Opening JSON file
    f = open(filename)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    f.close()
    
    return data