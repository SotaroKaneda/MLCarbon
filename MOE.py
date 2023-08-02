from model import Model

class MOE(Model):
    def __init__(self, parameters_b, tokens_t, percent_activated = 1, num_expert = 1, top_k = 1):
        super().__init__(parameters_b / num_expert, tokens_t, percent_activated)
        self.num_expert = num_expert
        self.top_k = top_k
        self.total_flops = self.total_flops * top_k
        return