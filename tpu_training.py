import csv
import json
from model import Model
from training import Training
import pandas as pd

class Tpu_train(Training):
    def __init__(self, model, version, num_tpu = 0):
        super().__init__()
        # load tpu info
        with open('data/tpus.json') as tpu_file:
            tpus = json.load(tpu_file)
        self.model = model
        self.tpu = tpus[version]
        self.num_tpu = num_tpu
        return
    
    def calc_train_time(self,):
        self.training_seconds = self.model.total_tflops / self.tflop_per_sec

    def get_training_hours(self):
        return self.training_seconds/ 60/ 60

    # tpu flop rate in flop per sec magnitude of E21
    def calc_tpu_tflops(self):
        # divide by billion to match units
        self.tflop_per_sec = self.num_tpu * self.tpu['peak TFLOPS'] * self.tpu['hardware_utilization']

    # calculate tflops using imperical hardware utilization
    def calc_tflops(self, hardware_efficiency):
        self.tflop_per_sec = self.num_tpu * self.tpu['peak TFLOPS'] * hardware_efficiency

    # energy in MWh
    def calc_energy(self,):
        # pod_energy in MWs
        pod_energy = self.num_tpu * self.tpu['power']['mean'] / 1e6
        self.total_energy = pod_energy * self.get_training_hours()

    # units of kg/kWh
    def get_co2(self, rate):
        return rate * self.total_energy

    # returns carbon emissions in kgs
    #def get_co2(self, provider, region):
        # impact is gCO2/kWh
    #    return impact(provider, region) / 1000 * self.total_energy

    
    # assuming tpu lifetime of 5 years
    def calc_embodied_carbon(self):
        self.embodied_carbon = self.get_pod_embodied() * self.get_training_hours() / (5 * 365 * 24)
    
    def get_pod_embodied(self,):
        with open('data/transistors.json') as chip_file:
            chips = json.load(chip_file)
        #  Carbon emitted Per unit Area
        brand = self.tpu['chip']['brand']
        size = self.tpu['chip']['size']
        tpu_carbon = self.num_tpu * self.tpu['die size'] * chips[brand][size]

        num_cpu = self.num_tpu / self.tpu['cpus per tpu']
        cpu_carbon = num_cpu * 0.7 * chips['TSMC']['7nm']
        return tpu_carbon + cpu_carbon

def impact(provider, region):
    impact_df = pd.read_csv('./data/impact.csv')
    datacenter = impact_df.loc[(impact_df['provider']==provider) & (impact_df['region']==region)]
    offset = datacenter["offsetRatio"]  # percentage of carbon emissions the provider offsets
    impact = datacenter["impact"]       # gCO2eq / kWh
    return impact.loc[datacenter.index[0]]