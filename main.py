import copy
import heapq
from typing import List

from fastapi import FastAPI
import numpy as np

from ai import Model, similarity
from data import Data
from entity import Sample, SearchSamplesRequestBody

app = FastAPI()
model: Model = Model()
data: Data = Data(model)


def extract(candidates: List[Sample], text_embedding: np.ndarray, num: int):
    temps = {}
    for i in range(len(candidates)):
        temps[i] = similarity(text_embedding, candidates[i].embedding)
    indices = heapq.nlargest(num, temps, temps.get)
    return [copy.deepcopy(candidates[i]) for i in indices]


@app.post("/search_samples")
async def search_samples(request_body: SearchSamplesRequestBody):
    if len(request_body.accusations) > 3:
        return "The length of accusations is over the limit"

    text_embedding: np.ndarray = model.embedding(request_body.text)

    results = []

    # 加载用户已选择的罪名条目；目标：搜索出相似度最高的 6 / len(accusation) 条
    if len(request_body.accusations) == 1:
        results += extract(data[request_body.accusations[0]], text_embedding, 6)
    elif len(request_body.accusations) == 2:
        for accusation_name_ in request_body.accusations:
            results += extract(data[accusation_name_], text_embedding, 3)
    elif len(request_body.accusations) == 3:
        for accusation_name_ in request_body.accusations:
            results += extract(data[accusation_name_], text_embedding, 2)

    # 加载用户未选择的罪名条目；目标：搜索出相似度最高的两条
    candidates = []
    for accusation_name_ in data:
        if accusation_name_ in request_body.accusations:
            continue
        candidates += data[accusation_name_]
    results += extract(candidates, text_embedding, 8 if len(request_body.accusations) == 0 else 2)

    # 去除 embedding 属性（无法 jsonify）
    for item in results:
        item.embedding = None

    return results
