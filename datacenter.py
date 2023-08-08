import csv
import pandas as pd
import json

# returns operational carbon emissions in tons
def co2_oper(impact, accelerator_energy):
    #impact is in kgCO2/kWh
    return impact / 1000 * accelerator_energy

# give a dictionary of name of device and the number
def co2_emb(devices, training_days):
  df = pd.read_csv('./data/hardware.csv')
  embodied_carbon = 0
  for d in devices:
    embodied_carbon += devices[d] * df.loc[df['hardware'] == d]['CO2e_chip(kgCO2e)'].values[0]
    #print(devices[d] * df.loc[df['hardware'] == d]['CO2e_chip(kgCO2e)'].values[0] / 1000)
  lifetime = 365 * 5
  return training_days / lifetime * embodied_carbon / 1000
  
def impact(provider, region):
  impact_df = pd.read_csv('./data/impact.csv')
  datacenter = impact_df.loc[(impact_df['provider']==provider) & (impact_df['region']==region)]
  offset = datacenter["offsetRatio"]  # percentage of carbon emissions the provider offsets
  impact = datacenter["impact"]       # gCO2eq / kWh
  return impact.loc[datacenter.index[0]]

def get_pod_embodied(self,):
    with open('data/transistors.json') as chip_file:
        chips = json.load(chip_file)
    #  Carbon emitted Per unit Area
    brand = self.tpu['chip']['brand']
    size = self.tpu['chip']['size']
    tpu_carbon = self.num * self.tpu['die size'] * chips[brand][size]

    num_cpu = self.num/ self.tpu['cpus per tpu']
    cpu_carbon = num_cpu * 0.7 * chips['TSMC']['7nm']
    return tpu_carbon + cpu_carbon
