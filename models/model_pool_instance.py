import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math

"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Max_Pool_Ins(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(Max_Pool_Ins, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h):
        h = self.feature(h)  # Nxsize[1]       
        
        logits_raw = self.classifiers(h) # Nx2 
        logits, idx = torch.max(logits_raw, dim= 0, keepdim= True) # 1x2
        A = torch.zeros(h.size()[0]).to(h.device)
        A[idx] = 1
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'attention': A}

        return logits, Y_prob, Y_hat, logits_raw, results_dict



"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Mean_Pool_Ins(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(Mean_Pool_Ins, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h):
        h = self.feature(h)  # Nxsize[1]       
                        
        logits_raw = self.classifiers(h) # Nx2
        logits = torch.mean(logits_raw, dim= 0, keepdim= True) # 1x2

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'attention': logits}

        return logits, Y_prob, Y_hat, logits_raw, results_dict
