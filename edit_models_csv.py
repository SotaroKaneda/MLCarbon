import pandas as pd
def models():
    models_df = pd.read_csv('./data/models.csv')
    tokens_t = [0.0099, 0.0099,4.009984,1,10,1.2,0.013924, 0]
    models_df.insert(2, 'Tokens(trillions)', tokens_t)
    models_df.to_csv('./data/models.csv', sep=',', index=False)

def table4():
    df = pd.read_csv('./data/table4.csv')
    t = df.transpose()
    t.to_csv('./data/table4.csv', sep=',', index=False)

table4()