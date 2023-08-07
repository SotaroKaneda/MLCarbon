import csv
import json
from model import Model
import pandas as pd
import numpy as np

class Tpu_train():
    def __init__(self, model, version, num = 0):
        super().__init__()
        # load tpu info
        with open('data/tpus.json') as tpu_file:
            tpus = json.load(tpu_file)
        self.model = model
        self.tpu = tpus[version]
        self.num = num
        return
    
    def calc_train_time(self,):
        self.training_seconds = self.model.total_tflops / self.tflop_per_sec

    def get_training_hours(self):
        return self.training_seconds/ 60/ 60
    
    def get_training_days(self):
        return self.training_seconds/ 60/ 60/ 24

    # tpu flop rate in flop per sec magnitude of E21
    def calc_tpu_tflops(self):
        # divide by billion to match units
        self.tflop_per_sec = self.num * self.tpu['peak TFLOPS'] * self.tpu['hardware_utilization']

    # calculate tflops using imperical hardware utilization
    def calc_tflops(self, hardware_efficiency):
        self.tflop_per_sec = self.num * self.tpu['peak TFLOPS'] * hardware_efficiency

    # energy in MWh
    def calc_energy(self,):
        # pod_energy in MWs
        cluster_energy = self.num * self.tpu['power']['mean'] / 1e6
        self.total_energy = cluster_energy * self.get_training_hours()

    # units of kg/kWh
    def get_co2(self, pue, carbon_rate):
        return self.total_energy * pue * carbon_rate

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
        tpu_carbon = self.num * self.tpu['die size'] * chips[brand][size]

        num_cpu = self.num/ self.tpu['cpus per tpu']
        cpu_carbon = num_cpu * 0.7 * chips['TSMC']['7nm']
        return tpu_carbon + cpu_carbon

    def predict_num_throughput(self, node_size):
        coeff_gpu = np.array([-2.12910565e-21,  4.39684339e-09,  7.99173057e+02])
        coeff_batch = np.array([-4.29439186e-01,  5.21376002e+01,  1.43737095e+03])
        func_gpu = np.poly1d(coeff_gpu)
        func_batch = np.poly1d(coeff_batch)
        gpu_cap = 0.03 * gpu['memory']
        if self.model.parameters_b < node_size * gpu_cap:
            p_size = 1
            t_size = int(np.ceil(self.model.parameters_b/gpu_cap))
        else:
            t_size = node_size
            p_size = int(np.ceil(self.model.parameters_b/(node_size*gpu_cap)))
        model_size = t_size * p_size
        num_gpu = np.round(func_gpu(self.model.parameters_b * 1e9) / model_size) * model_size
        if gpu['memory'] != 80:
            num_gpu = np.round(num_gpu * 2.5)
        if gpu['memory'] == 40:
            num_gpu *= 2
        d_size = num_gpu / model_size
        #estimated batch size
        if p_size == 1:
            batch_size = 512
        else:
            batch_size = np.round(func_batch(p_size)/8)*8
        if batch_size < num_gpu:
            batch_size = num_gpu
        coeff_tensor = np.array([-8.82079068e-20,  1.68591116e-09,  1.33954735e+02])
        coeff_pipe = np.array([-5.60233749e-23,  8.45435587e-11,  1.34546129e+02])
        func_tensor = np.poly1d(coeff_tensor)
        func_pipe = np.poly1d(coeff_pipe)
        rel_thru = 7.76 / 33.46
        #intra model condition
        if (t_size <= node_size and p_size == 1):
            X = func_tensor(self.model.parameters_b * 1e9)
        # inter model
        else:
            X = func_pipe(self.model.parameters_b * 1e9)
        if gpu['memory'] != 80:
            X_new = X -  X*rel_thru
            peak_new = X_new /312
            self.throughput= peak_new*125
        else:
            self.throughput = X
        self.model.num = num_gpu

def impact(provider, region):
    impact_df = pd.read_csv('./data/impact.csv')
    datacenter = impact_df.loc[(impact_df['provider']==provider) & (impact_df['region']==region)]
    offset = datacenter["offsetRatio"]  # percentage of carbon emissions the provider offsets
    impact = datacenter["impact"]       # gCO2eq / kWh
    return impact.loc[datacenter.index[0]]
