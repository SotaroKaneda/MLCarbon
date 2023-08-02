import csv
import json
from model import Model
import pandas as pd
import numpy as np

class Gpu_train:
    def __init__(self, model, gpu, num_gpu, node_size):
        gpu_df = pd.read_csv('./data/impact.csv')
        self.model = model
        with open('data/gpus.json') as gpu_file:
            gpus = json.load(gpu_file)
        self.gpu = gpus[gpu]
        self.num_gpu = num_gpu
        self.node_size = node_size
        return
    
    def get_ptd(self):
        gpu_cap = 0.03 * self.gpu['memory']
        if self.model.parameters_b < self.node_size * gpu_cap:
            p_size = 1
            t_size = int(np.ceil(self.model.parameters_b/gpu_cap))
        else:
            t_size = self.node_size
            p_size = int(np.ceil(self.model.parameters_b/(self.node_size*gpu_cap)))
        
        model_size = t_size * p_size

