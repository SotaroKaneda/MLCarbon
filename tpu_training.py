import csv
import json
from model import Model
from training import Training
import pandas as pd

class Tpu_train(Training):
    def __init__(self, model, version, num_tpu):
        super().__init__()
        # load tpu info
        with open('data/tpus.json') as tpu_file:
            tpus = json.load(tpu_file)
        self.model = model
        self.tpu = tpus[version]
        self.num_tpu = num_tpu
        self.calc_tpu_flops()
        self.calc_train_time()
        self.calc_energy()
        self.calc_embodied_carbon()
        return
    
    # tpu flop rate in flop per sec magnitude of E21
    def calc_tpu_flops(self):
        self.flop_per_sec = self.num_tpu * self.tpu['peak TFLOPS'] * self.tpu['hardware_utilization'] / 1000000000


    # energy in MWh
    def calc_energy(self,):
        # pod_energy in MWs
        pod_energy = self.num_tpu * self.tpu['power']['mean'] / 1000000
        self.energy = pod_energy * self.get_training_hours()

    # assuming tpu lifetime of 5 years
    def calc_embodied_carbon(self):
        self.embodied_carbon = self.get_pod_embodied() * self.get_training_hours() / (5 * 365 * 24)

       
    def get_(self):
        return self
    
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
