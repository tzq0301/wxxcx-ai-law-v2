import numpy as np
import unittest

from ai import Model, similarity


class AiTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Model = Model()

    def test_embedding(self):
        self.assertEqual(np.shape(self.model.embedding("我是真的牛")), (768,))

    def test_similarity(self):
        self.assertLessEqual(similarity(
            self.model.embedding("我是真的牛"),
            self.model.embedding("我是真的帅")), 1)


if __name__ == '__main__':
    unittest.main()
