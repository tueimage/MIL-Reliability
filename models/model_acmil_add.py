import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
        


"""
args:
    gate: whether to use gated attention
    size_arg: size of the model (small or big)
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    n_token: number of tokens
    n_masked_patch: number of patches to be masked
    mask_drop: percentage of patches to be masked
"""
class ACMIL_ADD(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, n_token=1, n_masked_patch=0, mask_drop=0.6):
        super(ACMIL_ADD, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_token)
        else:    
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_token)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifiers = nn.ModuleList()
        for i in range(n_token):
            self.classifiers.append(nn.Linear(size[1], n_classes))
            
        self.bag_classifier = nn.Linear(size[1], n_classes)
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.mask_drop = mask_drop
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.bag_classifier = self.bag_classifier.to(device)
        
    def forward(self, h, attention_only=False, use_attention_mask=False):
        A, h = self.attention_net(h)  # NxK  
              
        A = torch.transpose(A, 1, 0)  # KxN
        
        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)
        
        if attention_only:
            return A
        A_raw = A
            
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        outputs = []
        for i, head in enumerate(self.classifiers):
            outputs.append(head(M[i]))
        
        bag_A = F.softmax(A_raw, dim=1).mean(0, keepdim=True)
        bag_A = torch.transpose(bag_A, 1, 0)  # NxK

        bag_feat = bag_A*h
        logits_raw = self.bag_classifier(bag_feat) # Nx2
        return torch.stack(outputs, dim=0), logits_raw, A_raw.unsqueeze(0)
    