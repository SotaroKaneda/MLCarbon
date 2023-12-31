
import unittest
import pandas as pd
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from training import Training
from datacenter import co2_oper, co2_emb

# Validates 3 values in Table 4, total flops, training days, carbon emission
class validate_tables(unittest.TestCase):
    def test_table_4(self):
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
                training = Training(model, row['device'], row['chip #'])
                training.calc_tflops(row['hardware eff.'] / 100)
                training.calc_train_time()
                error = abs(training.get_training_days() - row['predicted days']) / row['predicted days'] * 100
                self.assertLess(error, 10, f'Prediction of Training Days of {row["LLM"]} is off by {error}%')

            # validate if Carbon emissions match
            with self.subTest('Validate Carbon', model = row['LLM']):
                training.calc_energy()
                co2_t = co2_oper(row['C02 e/KWh'], training.total_energy * row['PUE']) * 1000
                error = abs(co2_t- row['predicted tC02 e']) / row['predicted tC02 e'] * 100
                self.assertLess(error, 10, f'Prediction of Operating Emissions of {row["LLM"]} is off by {error}%')
    
    # validate time/ lifetime and CO2 emb
    def test_table_5(self):
        training_days = 20.4275
        df5 = pd.read_csv('./data/table5.csv')
        emb = 0
        for i in df5.index:
            row = df5.iloc[i]
            emb += co2_emb(row['hardware'], row['number'], training_days)
        error = abs(emb - 0.66)/ 0.66 * 100
        self.assertLess(error, 3, f'Prediction of Embodied Carbon of XLM is off by {error}%')


    
unittest.main()