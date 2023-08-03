import json
import csv
import numpy as np
class Model():
    def __init__(self, parameters_b, tokens_t, percent_activated = 100):
        self.parameters_b = parameters_b
        self.tokens_t = tokens_t
        self.activated_parameters_b = parameters_b * percent_activated / 100
        # source for formula https://arxiv.org/pdf/2203.15556.pdf
        # in units of 1e9 * 1e12 = 1e21
        self.total_tflops = 6 * self.activated_parameters_b * tokens_t * 1e9
        return
    
    # update the size and test loss after sparseGPT unstructured 50% pruning
    # unstructured pruning is benificial for CPU inference of small models
    # https://arxiv.org/pdf/2301.00774.pdf "CPU speedups"
    def unstructured_50_sparsity(self):
        size = self.parameters_b
        # initialize for now becuase we are not concerned of perplexity when calculating carbon emissions
        perplexity = 10
        dense_perplexity = np.array([12.47,10.86,10.13, 9.56,9.34,8.35])
        sparse_perplexity = np.array([13.48,11.55,11.17,9.79,9.32,8.21])
        y = dense_perplexity - sparse_perplexity # improvement
        x = model_size = np.array([2.7, 6.7, 13, 30, 66, 175]) # in billions
        improvement = np.poly1d(np.polyfit(x, y, 3))
        # return the speedup on CPU and new perplexity
        return [1.82, perplexity - improvement(size)]
    
    # sparseGPT structured 2:4 pruning
    # structured pruning is benificial for NVIDIA GPUs
    # https://arxiv.org/pdf/2301.00774.pdf "GPU speedups"
    def structured_2_4_sparsity(self):
        size = self.parameters_b
        # initialize for now becuase we are not concerned of perplexity when calculating carbon emissions
        perplexity = 10
        dense_perplexity = np.array([12.47,10.86,10.13, 9.56,9.34,8.35])
        sparse_perplexity = np.array([17.18,14.2,12.96,10.9,10.09,8.74])
        y = dense_perplexity - sparse_perplexity # improvement
        x = model_size = np.array([2.7, 6.7, 13, 30, 66, 175]) # in billions
        improvement = np.poly1d(np.polyfit(x, y, 3))
        # We use speedup for Feed Forward Layer because of its higher latency
        return [1.67, perplexity - improvement(size)]

    