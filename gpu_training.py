import csv

import json
from model import Model
from training import Training
import pandas as pd
import numpy as np

class Gpu_train(Training):
    def __init__(self, model, gpu, num_gpu = 1, node_size = 1):
        super().__init__()
        gpu_df = pd.read_csv('./data/impact.csv')
        self.model = model
        with open('data/gpus.json') as gpu_file:
            gpus = json.load(gpu_file)
        self.gpu = gpus[gpu]
        self.num_gpu = num_gpu
        self.node_size = node_size
        self.get_ptd()
        self.get_throughput()
        self.calc_train_time()
        self.calc_energy()
        return
    

    def get_ptd(self):
        coeff_gpu = np.array([-2.12910565e-21,  4.39684339e-09,  7.99173057e+02])
        coeff_batch = np.array([-4.29439186e-01,  5.21376002e+01,  1.43737095e+03])
        func_gpu = np.poly1d(coeff_gpu)
        func_batch = np.poly1d(coeff_batch)
        gpu_cap = 0.03 * self.gpu['memory']
        if self.model.parameters_b < self.node_size * gpu_cap:
            self.p_size = 1
            self.t_size = int(np.ceil(self.model.parameters_b/gpu_cap))
        else:
            self.t_size = self.node_size
            self.p_size = int(np.ceil(self.model.parameters_b/(self.node_size*gpu_cap)))
        
        model_size = self.t_size * self.p_size

        self.num_gpu = np.round(func_gpu(self.model.parameters_b * 1e9) / model_size) * model_size
        if self.gpu['memory'] != 80:
            self.num_gpu = np.round(self.num_gpu * 2.5)
        if self.gpu['memory'] == 40:
            self.num_gpu *= 2
        
        self.d_size = self.num_gpu / model_size
        #estimated batch size
        if self.p_size == 1:
            self.batch_size = 512
        else:
            self.batch_size = np.round(func_batch(self.p_size)/8)*8
        if self.batch_size < self.num_gpu:
            self.batch_size = self.num_gpu
        return

    def get_throughput(self):
        coeff_tensor = np.array([-8.82079068e-20,  1.68591116e-09,  1.33954735e+02])
        coeff_pipe = np.array([-5.60233749e-23,  8.45435587e-11,  1.34546129e+02])
        func_tensor = np.poly1d(coeff_tensor)
        func_pipe = np.poly1d(coeff_pipe)
        rel_thru = 7.76 / 33.46
        #intra model condition
        if (self.t_size <= self.node_size and self.p_size == 1):
            X = func_tensor(self.model.parameters_b * 1e9)
        # inter model
        else:
            X = func_pipe(self.model.parameters_b * 1e9)

        if self.gpu['memory'] != 80:
            X_new = X -  X*rel_thru
            peak_new = X_new /312
            self.throughput= peak_new*125
        else:
            self.throughput = X
        self.tflop_per_sec = self.num_gpu * self.throughput
        return

    def calc_energy(self):
        self.total_energy = self.gpu['tdp'] * self.get_training_hours()
        
gpt_model = Model(175, 0.3)
training = Gpu_train(gpt_model, 'V100', 1, 8)
print(training.model.total_tflops)
print(training.throughput)
print(training.num_gpu)
print(training.get_training_days())