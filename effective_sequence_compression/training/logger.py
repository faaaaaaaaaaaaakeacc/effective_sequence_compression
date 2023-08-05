import time
import json
import os
import torch


class Logger:
    def __init__(self):
        self.losses = []
        self.metrics = []
        self.times_losses = []
        self.times_metrics = []
        self.model = None
    
    def log_loss(self, val):
        self.losses.append(val)
        self.times_losses.append(time.time())
        
    def log_metric(self, val):
        self.metrics.append(val)
        self.times_metrics.append(time.time())

    def set_model(self, model):
        self.model = model

    def save(self, filename):
        info_logger = {
            "losses" : self.losses,
            "metrics": self.metrics,
            "times_losses" : self.times_losses,
            "times_metrics" : self.times_metrics
        }
        if self.model is not None:
            torch.save(self.model.state_dict(), f"model_{filename}")
            info_logger['model_truncation_list'] = self.model.truncation_list
        with open(filename, 'w+') as convert_file:
            convert_file.write(json.dumps(info_logger))


    def upload(self, filename):
        f = open(filename)
        data = json.load(f)
        return data
