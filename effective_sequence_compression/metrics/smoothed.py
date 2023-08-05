class SmoothedMetrics:
    def __init__(self, metrics_list, metric_names):
        self.metrics_list = metrics_list
        self.metrics_names = metric_names
        
    def __call__(self, preds, targets):
        for metric in self.metrics_list:
            metric(preds, targets)
            
    def finish(self):
        answer = {}
        for name, metric in zip(self.metrics_names, self.metrics_list):
            answer[name] = metric.finish()
        return answer