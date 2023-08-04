from datasets import load_dataset
from effective_sequence_compression.datasets.batch import Batch

class SentimentDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = load_dataset("sentiment140")
        self.max_target = None
        
    def iter_batches_train(self):
        for i in range(len(self.dataset['train']) // self.batch_size):
            text_features = []
            targets = []
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                elem = self.dataset['train'][j]
                targets.append(elem['sentiment'])
                feature = f"TEXT: {elem['text']} DATE: {elem['date']} USER: {elem['user']} QUERY: {elem['query']}"
                text_features.append(feature)
            yield Batch(text_features, targets)
    
    def iter_batches_test(self):
        for i in range(len(self.dataset['test']) // self.batch_size):
            text_features = []
            targets = []
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                elem = self.dataset['test'][j]
                targets.append(elem['sentiment'])
                feature = f"TEXT: {elem['text']} DATE: {elem['date']} USER: {elem['user']} QUERY: {elem['query']}"
                text_features.append(feature)
            yield Batch(text_features, targets)

    def get_num_targets(self):
        if self.max_target is not None:
            return self.max_target
        self.max_target = 0
        for i in range(len(self.dataset['train'])):
            self.max_target = max(self.max_target, self.dataset['train'][i]['sentiment'])
        return self.max_target
