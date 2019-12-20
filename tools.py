import os
import time
import torch.nn.functional as F
import torch 
import numpy as np

def one_hot_encode(num_classes, class_idx):
    return torch.eye(num_classes)[class_idx]

def label2edge(label):
    ns = len(label)
    label = label.unsqueeze(-1).repeat(1, ns)
    edge = torch.eq(label, label.transpose(0,1)).float()
    return edge

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
