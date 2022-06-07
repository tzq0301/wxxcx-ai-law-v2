import numpy as np
import unittest

from data import Data
from ai import Model


class DataTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Model = Model()
        self.data: Data = Data(self.model)

    def test_get_data(self):
        self.assertEqual(np.shape(self.data["失火"][0].embedding), (768, ))  # add assertion here


if __name__ == '__main__':
    unittest.main()
