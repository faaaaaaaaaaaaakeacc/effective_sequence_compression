from sklearn.metrics import roc_auc_score
import numpy as np


class AccuracyWithoutThresholdBinary:
    def __init__(self):
        self.predictions = []
        self.targets = []
        
    def __call__(self, preds, target):
        assert len(preds) == len(target)
        for i in range(len(preds)):
            self.predictions.append(preds[i])
            self.targets.append(target[i])
            
    def finish(self):
        paired_info = sorted(list(zip(self.predictions, self.targets)))
        cnt_positive = sum(self.targets)
        cnt_negative = 0
        max_acc = 0
        for i in range(len(paired_info) + 1):
            max_acc = max(max_acc, (cnt_negative + cnt_positive) / len(self.targets))
            if i < len(paired_info) and paired_info[i][1] == 0:
                cnt_negative += 1
            else:
                cnt_positive -= 1
        self.predictions = []
        self.targets = []

        return max_acc


class F1WithoutThresholdBinary:
    def __init__(self):
        self.predictions = []
        self.targets = []
        
    def __call__(self, preds, target):
        assert len(preds) == len(target)
        for i in range(len(preds)):
            self.predictions.append(preds[i])
            self.targets.append(target[i])
        
    def finish(self):
        paired_info = sorted(list(zip(self.predictions, self.targets)))
        tp, fp, tn, fn = sum(self.targets), len(self.targets) - sum(self.targets), 0, 0
        max_f1 = 0
        for i in range(len(self.targets) + 1):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)
            if i < len(self.targets) and paired_info[i][1] == 1:
                fn += 1
                tp -= 1
            else:
                tn += 1
                fp -= 1
        self.predictions = []
        self.targets = []

        return max_f1 


class RocAucBinary:
    def __init__(self):
        self.predictions = []
        self.targets = []
        
    def __call__(self, preds, target):
        assert len(preds) == len(target)
        for i in range(len(preds)):
            self.predictions.append(preds[i])
            self.targets.append(target[i])

    def finish(self):
        out = roc_auc_score(self.targets, self.predictions) 
        self.predictions = []
        self.targets = []
        return out


def target_to_one_hot(target, num_classes):
    answer = np.zeros((len(target), num_classes))
    for i in range(len(target)):
        answer[i][target[i]] = 1
    return answer


class AccuracyWithoutThresholdsMacro:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.metrics = [AccuracyWithoutThresholdBinary() for _ in range(num_classes)]

    def __call__(self, preds, target):
        assert len(preds) == len(target)
        one_hot_target = target_to_one_hot(target, self.num_classes)
        for i in range(self.num_classes):
            self.metrics[i](preds[:, i], one_hot_target[:, i])
    
    def finish(self):
        arr = [self.metrics[i].finish() for i in range(self.num_classes)]
        return np.mean(arr)


class F1WithoutThresholdsMacro:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.metrics = [F1WithoutThresholdBinary() for _ in range(num_classes)]

    def __call__(self, preds, target):
        assert len(preds) == len(target)
        one_hot_target = target_to_one_hot(target, self.num_classes)
        for i in range(self.num_classes):
            self.metrics[i](preds[:, i], one_hot_target[:, i])
    
    def finish(self):
        arr = [self.metrics[i].finish() for i in range(self.num_classes)]
        return np.mean(arr)


class RocAucMacro:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.metrics = [RocAucBinary() for _ in range(num_classes)]

    def __call__(self, preds, target):
        assert len(preds) == len(target)
        one_hot_target = target_to_one_hot(target, self.num_classes)
        for i in range(self.num_classes):
            self.metrics[i](preds[:, i], one_hot_target[:, i])
    
    def finish(self):
        arr = [self.metrics[i].finish() for i in range(self.num_classes)]
        return np.mean(arr)
