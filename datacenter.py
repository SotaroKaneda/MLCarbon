import csv
import pandas as pd

def impact(provider, region):
  impact_df = pd.read_csv('./data/impact.csv')
  datacenter = impact_df.loc[(impact_df['provider']==provider) & (impact_df['region']==region)]
  offset = datacenter["offsetRatio"]  # percentage of carbon emissions the provider offsets
  impact = datacenter["impact"]       # gCO2eq / kWh
  return impact.loc[datacenter.index[0]]

# returns carbon emissions in kgs
def calc_co2(provider, region, accelerator_energy):
    #impact is gCO2/kWh
    return impact(provider, region) / 1000 * accelerator_energy