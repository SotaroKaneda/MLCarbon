
import unittest
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from gpu_training import Gpu_train
import numpy as np


class Gpu_test(unittest.TestCase):
    def test_GPT_on_8x_80GB_A100(self):
        gpt_model = Model(175, 0.3)
        training = Gpu_train(gpt_model, 'A100', 1, 8)
        original_energy = 156
        self.assertEqual(original_energy, np.ceil(training.total_energy), "doesn't match original analysis")

unittest.main()