# 基于 transformers 的法条文本相似度匹配

## 运行

```shell
uvicorn main:app --reload
```

> The command uvicorn main:app refers to:
> - `main`: the file main.py (the Python "module").
> - `app`: the object created inside of main.py with the line app = FastAPI().
> - `--reload`: make the server restart after code changes. Only do this for development.

## 依赖

### FastAPI

[FastAPI](https://fastapi.tiangolo.com/)

### SentenceTransformers

[SentenceTransformers](https://www.sbert.net/)

### ONNX & ONNX Runtime

[ONNX Python](https://onnxruntime.ai/docs/get-started/with-python.html)

### HuggingFace & symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli

Model: [symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli](https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli)

[Export HuggingFace Models](https://huggingface.co/docs/transformers/serialization#exporting-a-model-to-onnx)