from model import Model

class NAS(Model):
    def __init__(self, parameters_b, tokens_t, percent_activated = 1, num_model = 1):
        super().__init__(parameters_b, tokens_t, percent_activated)
        self.num_model = num_model
        self.total_flops = self.total_flops * num_model
        return