
import unittest
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from gpu_training import Gpu_train


class Gpu_test(unittest.TestCase):
    def test_GPT_on_8x_80GB_A100(self):
        gpt_model = Model(175, 0.3)
        training = Gpu_train(gpt_model, 'A100', 1, 8)
        original_energy = 155978.0
        self.assertEqual(155978.0, training.total_energy, "doesn't match original analysis")

unittest.main()