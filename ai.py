import numpy as np
import onnx
from onnxruntime import InferenceSession
from scipy import spatial
from transformers import AutoTokenizer


class Model:
    def __init__(self,
                 onnx_model_path: str = "static/model.onnx",
                 tokenizer_name: str = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"):
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._session = InferenceSession(onnx_model_path)

    def embedding(self, text: str) -> np.ndarray:
        """
        对文本进行句嵌入

        :param text: 文本
        :return: 一维 numpy.ndarray 向量（长度为 768）
        """
        inputs = self._tokenizer(text, return_tensors="np")  # ONNX Runtime expects NumPy arrays as input
        outputs = self._session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
        return outputs[0][-1, -1, :]  # (768, )


def similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    基于 scipy 计算两个 np.ndarray 变量的余弦相似度
    """
    return 1 - spatial.distance.cosine(x, y)
