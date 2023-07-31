import csv
import pandas as pd

# returns carbon emissions in kgs
def calc_co2(region, accelerator_energy):
    datacenter_df = pd.read_csv('./data/impact.csv')
    region_row = datacenter_df.loc[datacenter_df['region'] == region]
    #impact is gCO2/kWh
    return region_row['impact'].values[0] / 1000 * accelerator_energy
