
import unittest
import pandas as pd
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from tpu_training import Tpu_train
import datacenter
# Validates 3 values in Table 4, total flops, training days, carbon emission
class validate_table4(unittest.TestCase):
    def test_CO2(self):
        df4 = pd.read_csv('./data/table4.csv')
        for i in df4.index:
            row = df4.iloc[i]

            # validate if flops match
            with self.subTest('Validate Total FLOPS', model = row['LLM']):
                model = Model(row['param. # (B)'], row['token # (B)'] / 1000)
                error = abs(model.total_tflops * 1e12 - row['predicted FLOPs']) / row['predicted FLOPs'] * 100
                self.assertLess(error, 10, f'Prediction of total FLOPS of {row["LLM"]} is off by {error}%')

            # validate if training time match
            with self.subTest('Validate Training Days', model = row['LLM']):
                training = Tpu_train(model, row['device'], row['chip #'])
                training.calc_tflops(row['hardware eff.'] / 100)
                training.calc_train_time()
                error = abs(training.get_training_days() - row['predicted days']) / row['predicted days'] * 100
                self.assertLess(error, 10, f'Prediction of Training Days of {row["LLM"]} is off by {error}%')

            # validate if Carbon emissions match
            with self.subTest('Validate Carbon', model = row['LLM']):
                training.calc_energy()
                error = abs(training.get_co2(row['PUE'], row['C02 e/KWh'])- row['predicted tC02 e']) / row['predicted tC02 e'] * 100
                self.assertLess(error, 10, f'Prediction of Operating Emissions of {row["LLM"]} is off by {error}%')
    
unittest.main()