import unittest
import pandas as pd
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from tpu_training import Tpu_train

# Test energy consumptions for all models in Carbon Emissions Paper
class Tpu_test(unittest.TestCase):
    def test_models(self):
        models_df = pd.read_csv('./data/models.csv')
        for i in models_df.index:
            row = models_df.iloc[i]
            with self.subTest(model = row['Model']):
                model = Model(row['Number of Parameters (B)'], row['Tokens(trillions)'], row['Percent of model activated on every token'])
                training = Tpu_train(model, row['Processor'], row['Number of Chips'])
                print(training.model.total_tflops)
                print(training.tflop_per_sec)
                energy = row['Energy Consumption (MWh)']
                error = (training.total_energy - energy) / energy  
                self.assertLess(abs(error), 0.5, "more than 50 error")

unittest.main()