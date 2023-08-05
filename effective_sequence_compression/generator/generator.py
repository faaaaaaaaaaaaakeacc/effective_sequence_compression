import random
import numpy as np


def scale(l, r, x):
    return min(r, max(l, x))


class UniformGenerator:
    def __init__(self, n_layers, max_length):
        self.n_layers = n_layers
        self.max_length = max_length
        
    def _sample(self, max_len, i):
        return scale(1, self.max_length, np.random.uniform(1, max_len * 6, 1)[0])

    def generate_sequence(self):
        answer = [self.max_length]
        for i in range(1, self.n_layers - 1):
            length = self._sample(self.max_length, self.n_layers - i)
            answer.append(length)
        answer.append(1)
        for i in range(len(answer)):
            answer[i] = min(answer[:(i + 1)])
        return [elem / self.max_length for elem in answer]
    
    def calc_V(self, path):
        score = 0
        for elem in path:
            score += elem * elem
        return score / self.n_layers

    def generate_with_rejection(self, left_V, right_V):
        while True:
            seq = self.generate_sequence()
            V = self.calc_V(seq)
            if left_V <= V and V <= right_V:
                return seq

    
class NormalGenerator:
    def __init__(self, n_layers, max_length):
        self.n_layers = n_layers
        self.max_length = max_length
        
    def _sample(self, max_len, i):
        return scale(1, self.max_length, np.random.normal(max_len * 1.2, max_len//3, 1)[0])

    def generate_sequence(self):
        answer = [self.max_length]
        for i in range(1, self.n_layers - 1):
            length = self._sample(self.max_length, self.n_layers - i)
            answer.append(length)
        answer.append(1)
        for i in range(len(answer)):
            answer[i] = min(answer[:(i + 1)])
        return [elem / self.max_length for elem in answer]
    
    def calc_V(self, path):
        score = 0
        for elem in path:
            score += elem * elem
        return score / self.n_layers

    def generate_with_rejection(self, left_V, right_V):
        while True:
            seq = self.generate_sequence()
            V = self.calc_V(seq)
            if left_V <= V and V <= right_V:
                return seq
