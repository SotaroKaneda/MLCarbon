
import unittest
import pandas as pd
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model

class Model_test(unittest.TestCase):
    def test_model_file(self):
        models_df = pd.read_csv('./data/models.csv')
        for i in models_df.index:
            row = models_df.iloc[i]
            with self.subTest(model = row['Model']): 
                print(f'TESTING: Total TFLOPs estimates for {row["Model"]}')
                model = Model(row['Number of Parameters (B)'], row['Tokens(trillions)'], row['Percent of model activated on every token'])
                total_flops = row['Total Computation (TFLOP)']
                error = (total_flops - model.total_tflops) / total_flops
                self.assertLess(abs(error), 1, "more than 100 percent error")
                print('SUCCESS\n')

unittest.main()