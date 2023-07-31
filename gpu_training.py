import csv
import json
from model import Model
import pandas as pd

class Gpu_train:
    def __init__(self, model, gpu, num_gpu):
        gpu_df = pd.read_csv('./data/impact.csv')
        self.model = model
        self.gpu_flops = 
        self.num_gpu = num_gpu
        return
