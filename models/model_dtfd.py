import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights

class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=256, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, L//2, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        initialize_weights(self)
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)    
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)
        initialize_weights(self)
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)  
        if self.droprate != 0.0:
            self.dropout = self.dropout.to(device)
    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512):
        super(DimReduction, self).__init__()
        fc = [nn.Linear(n_channels, m_dim), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.fc= nn.Sequential(*fc)
        initialize_weights(self)
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)  
    def forward(self, x):
        x = self.fc(x)
        return x


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attention(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attention, self).__init__()
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
        initialize_weights(self)
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_a = self.attention_a.to(device)
        self.attention_b = self.attention_b.to(device)
        self.attention_c = self.attention_c.to(device)
    def forward(self, x, isNorm=True):
        ## x: N x L
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A ### K x N