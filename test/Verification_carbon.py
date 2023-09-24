import pandas as pd

from model import Test_Model

models_df = pd.read_csv('./data/Model_detail.csv')
model_choice = int(input("Please choose LLM model:\n1.T5\n2.GPT3\n3.Gshard\n4.Switch\n5.XLM\n")) - 1
model_row = models_df.iloc[model_choice]

model = Test_Model(model_row['LLM'], model_row['type'],
                   model_row['parameter # (B)'], model_row['base model param. # (B)'], model_row['token # (B)'],
                   model_row['CO2eq/KWh '],
                   model_row['achieved TFLOPs/s'],
                   model_row['device #'],
                   model_row['avg. system power (W)'],
                   model_row['PUE'])
train_day = model.cal_trainday()
train_energy = model.pre_energy()
train_carbon = model.pre_carbon()
print("Model Name:{}\nModel Type:{}\nTraining Days:{:.2f}\nTraining Energy:{:.4f}\nTraining Carbon:{:.4f}".format(
    model_row['LLM'], model_row['type'], train_day, train_energy, train_carbon))
