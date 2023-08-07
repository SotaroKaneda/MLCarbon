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


def hardware_add_CO2chip():
    df = pd.read_csv('./data/hardware.csv')
    co2 = []
    for i in df.index:
        row = df.iloc[i]
        unit = row['unit']
        u = int(''.join(filter(str.isdigit, unit)))
        cpa = row['CPA']
        c = int(''.join(filter(str.isdigit, cpa)))
        print(u)
        print(c)
        if row['hardware'] == 'DRAM':
            co2.append(u * c)
        elif row['hardware'] == 'SSD':
            co2.append(u * 1000 * c)
        else:
            co2.append(u / 100 * c)
        print(co2)
        #df.insert(4, 'CO2e_chip(kgCo2)', co2)
    df.to_csv('./data/hardware.csv', sep=',', index=False)


hardware_add_CO2chip()