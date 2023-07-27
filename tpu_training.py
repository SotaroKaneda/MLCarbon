import csv
import json

class Tpu_train:
    def __init__(self, model, version, num_tpu):
        # load tpu info
        with open('data/tpus.json') as tpu_file:
            tpus = json.load(tpu_file)
        self.tpu = tpus[version]
        print(self.tpu)
        self.num_tpu = num_tpu
        return

    def calc_tpu_flops(self):
        self.pod_flops = self.num_tpu * self.tpu['max_flops'] * self.tpu['util_rate']

    def calc_train_time(self):
        self.training_seconds = self.model.total_flops / self.pod_flops

    def calc_energy(self):
        pod_energy = self.num_tpu * self.tpu['mean_watts']
        self.energy = pod_energy
    
    def get_training_hours(self):
        return self.training_seconds/ 60/ 60
    
    def get_training_days(self):
        return self.training_seconds/ 60/ 60/ 24
    
    def get_(self):
        return self
    


tpu = Tpu_train('asdf', 'TPUv1', 'adsf')