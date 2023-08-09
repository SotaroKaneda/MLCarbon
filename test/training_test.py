import unittest
import pandas as pd
import sys
sys.path.insert(0, '/Users/sotarokaneda/git/MLCarbon')
from model import Model
from training import Training
import numpy as np

# Test energy consumptions for all models in Carbon Emissions Paper
class Training_test(unittest.TestCase):
    def test_models(self):
        print("TESTING: Carbon Emissions Table Results")
        models_df = pd.read_csv('./data/models.csv')
        for i in models_df.index:
            row = models_df.iloc[i]
            with self.subTest(model = row['Model']):
                print(f'TESTING: Energy estimates for {row["Model"]}')
                model = Model(row['Number of Parameters (B)'], row['Tokens(trillions)'], row['Percent of model activated on every token'])
                training = Training(model, row['Processor'], row['Number of Chips'])
                training.calc_tpu_tflops()
                training.calc_train_time()
                training.calc_energy()
                energy = row['Energy Consumption (MWh)']
                error = (training.total_energy - energy) / energy  
                self.assertLess(abs(error), 0.5, "more than 50 error")
                print("SUCCESS\n")

    #Testing megatronLM paper Table 1 results
    def test_megatronLM(self):
        print("TESTING: Metagron Table 1 Results")
        megatron_results = pd.read_csv('./data/megatron_table.csv')
        for i in megatron_results.index:
            trial = megatron_results.iloc[i]
            with self.subTest(parameter_b = trial['Number of parameters (billion)']):
                model = Model(trial['Number of parameters (billion)'], 0.3)
                training = Training(model, 'A100')
                training.predict_num_throughput(8)
                table_throughput = trial['Achieved teraFLOP/s per GPU']
                print(f'TESTING: Throughput estimates for {trial["Number of parameters (billion)"]} billion parameter model')
                difference = abs(training.tflop_per_sec - table_throughput)
                self.assertLess(difference, 1, "more than 0.5 tFLOP/s error")
                print("SUCCESS\n")

    #Testing analysis data on CO2EQ jupyternotebook
    def test_GPT_on_8x_80GB_A100(self):
        print("TESTING: Throughput estimates for 8x 80GB A100 nodes")
        gpt_model = Model(175, 0.3)
        training = Training(gpt_model, 'A100')
        training.predict_num_throughput(8)
        training.calc_train_time()
        original_days = 17
        self.assertEqual(original_days, np.ceil(training.get_training_days()), "doesn't match original analysis")
        print("SUCCESS\n")

    #Testing analysis data on CO2EQ jupyternotebook
    def test_GPT_on_8x_32GB_V100(self):
        print("TESTING: Throughput estimates for 8x 32GB V100 nodes")
        gpt_model = Model(175, 0.3)
        training = Training(gpt_model, 'V100')
        training.predict_num_throughput(8)
        training.calc_train_time()
        original_days = 22 
        self.assertEqual(original_days, np.ceil(training.get_training_days()), "doesn't match original analysis")
        print("SUCCESS\n")

unittest.main()