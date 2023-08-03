
import unittest
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from gpu_training import Gpu_train
import numpy as np
import pandas as pd


class Gpu_test(unittest.TestCase):

    def test_megatronLM(self):
        print("TESTING: Metagron Table 1 results")
        megatron_results = pd.read_csv('./data/megatron_table.csv')
        for i in megatron_results.index:
            trial = megatron_results.iloc[i]
            with self.subTest(parameter_b = trial['Number of parameters (billion)']):
                model = Model(trial['Number of parameters (billion)'], 0.3)
                training = Gpu_train(model, 'A100', 1, 8)
                table_throughput = trial['Achieved teraFLOP/s per GPU']
                print(f'TESTING: Throuput estimates for {trial["Number of parameters (billion)"]} billion parameter model')
                difference = abs(training.throughput - table_throughput)
                self.assertLess(difference, 1, "more than 0.5 tFLOP/s error")
                print("SUCCESS\n")

    def test_GPT_on_8x_80GB_A100(self):
        print("TESTING: Energy estimates for 8x 80GB A100 nodes")
        gpt_model = Model(175, 0.3)
        training = Gpu_train(gpt_model, 'A100', 1, 8)
        original_energy = 156
        self.assertEqual(original_energy, np.ceil(training.total_energy), "doesn't match original analysis")
        print("SUCCESS\n")

    def test_GPT_on_8x_32GB_V100(self):
        print("TESTING: Energy estimates for 8x 32GB V100 nodes")
        gpt_model = Model(175, 0.3)
        training = Gpu_train(gpt_model, 'V100', 1, 8)
        original_energy = 158
        self.assertEqual(original_energy, np.ceil(training.total_energy), "doesn't match original analysis")
        print("SUCCESS\n")

unittest.main()