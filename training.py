from model import Model

class Training:
    def __init__(self):
        return
    
    def calc_train_time(self,):
        self.training_seconds = self.model.total_flops / self.flop_per_sec
    def get_training_hours(self):
        return self.training_seconds/ 60/ 60
    
    def get_training_days(self):
        return self.training_seconds/ 60/ 60/ 24
