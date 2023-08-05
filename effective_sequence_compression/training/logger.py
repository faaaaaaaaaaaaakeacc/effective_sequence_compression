import time
import json
import os

class Logger:
    def __init__(self):
        self.losses = []
        self.metrics = []
        self.times_losses = []
        self.times_metrics = []
    
    def log_loss(self, val):
        self.losses.append(val)
        self.times_losses.append(time.time())
        
    def log_metric(self, val):
        self.metrics.append(val)
        self.times_metrics.append(time.time())

    def save(self, filename):
        info_logger = {
            "losses" : self.losses,
            "metrics": self.metrics,
            "times_losses" : self.times_losses,
            "times_metrics" : self.times_metrics
        }
        with open(filename, 'w+') as convert_file:
            convert_file.write(json.dumps(info_logger))


    def upload(self, filename):
        f = open(filename)
        data = json.load(f)
        return data
