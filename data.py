from collections import defaultdict
import json
from tqdm import tqdm

from ai import Model
from entity import Sample


class Data:
    def __init__(self, model: Model, data_path: str = "static/final_data.txt"):
        self._data = defaultdict(list)

        with open(data_path, "r") as f:
            for line in tqdm(f.readlines(), desc="加载数据"):
                obj = json.loads(line)
                obj["embedding"] = model.embedding(obj["fact"][:512])
                sample = Sample(**obj)
                for accusation in sample.meta.accusation:
                    self._data[accusation].append(sample)

    def __getitem__(self, accusation: str):
        return self._data[accusation]

    def __iter__(self):
        return self._data.__iter__()
