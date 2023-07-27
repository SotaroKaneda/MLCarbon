import json
import csv

class Model():
    def __init__(self, parameters, tokens, percent_activated = 1):
        self.activated_parameters = parameters * percent_activated
        # source for formula https://arxiv.org/pdf/2203.15556.pdf
        self.total_flops = 6 * parameters * tokens
        return
    