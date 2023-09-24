import pandas as pd

from model import Dense, MoE

Type_choice = int(input("Please choose LLM type:\n1.Dense\n2.MoE\n"))
pre_param = 0
if Type_choice == 1:
    models_df = pd.read_csv('./data/Dense_model.csv')
    model_choice = int(input("Please choose LLM model:\n1.T5\n2.GPT3\n3.XLM\n4.Noor\n5.PaLM\n6.Gopher"
                             "\n7.Chinchilla\n8.LaMDA\n9.Jurassic-1\n10.MT-NLG\n11.Bloom\n12.YaLM\n13.GLM\n")) - 1
    model_row = models_df.iloc[model_choice]
    model = Dense(model_row['model_name'], model_row['hidden_size'],
                  model_row['ff_size'], model_row['head_size'], model_row['head_num'], model_row['layer_num'],
                  model_row['vocab_size'])
    pre_param = model.pre_param()
if Type_choice == 2:
    models_df = pd.read_csv('./data/MoE_model.csv')
    model_choice = int(
        input("Please choose LLM model:\n1.Gshard\n2.Switch\n3.GLaM\n4.FacebookMoE\n5.ST-MoE\n6.PR-MoE\n")) - 1
    model_row = models_df.iloc[model_choice]
    model = MoE(model_row['model_name'], model_row['base_param'], model_row['MoE'], model_row['hidden_size'],
                model_row['ff_size'], model_row['head_size'], model_row['layer_num'], model_row['expert_num'],
                model_row['head_num'])
    pre_param = model.pre_param()
print("Predict Parameter:{:.4f}".format(pre_param))
