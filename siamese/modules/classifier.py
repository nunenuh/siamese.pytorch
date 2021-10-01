import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Classfiers(nn.Module):
    def __init__(self, in_features: int, n_classes: int, 
                 use_batchnorm: bool = True, use_dropout: bool = True, 
                 dprob: List[float] = [0.5,0.3,0.2], **kwargs):
        super(Classfiers, self).__init__()
        modules = []
        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features))
        if use_dropout: modules.append(nn.Dropout(dprob[0]))
        modules.append(nn.Linear(in_features, in_features // 2))
        modules.append(nn.ReLU(inplace=True))

        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features//2))
        if use_dropout: modules.append(nn.Dropout(dprob[1]))
        modules.append(nn.Linear(in_features //2, in_features // 4))
        modules.append(nn.ReLU(inplace=True))

        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features//4))
        if use_dropout: modules.append(nn.Dropout(dprob[2]))
        modules.append(nn.Linear(in_features //4, n_classes))

        self.classfiers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        x = self.classfiers(x)
        return x


class SimpleClassifiers(nn.Module):
    def __init__(self, in_features: int, n_classes: int, 
                 use_batchnorm: bool = True, use_dropout: bool = True, 
                 dprob: float = 0.3, **kwargs):
        super(SimpleClassifiers, self).__init__()
        if use_batchnorm: self.bn = nn.BatchNorm1d(in_features)
        if use_dropout: self.dropout = nn.Dropout(p=dprob)
        self.fc = nn.Linear(in_features, n_classes)

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

    def forward(self, x: torch.Tensor):
        if self.use_batchnorm: x = self.bn(x)
        if self.use_dropout: x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
def simple_squential_classifier(in_features: int, n_classes: int, 
                                use_batchnorm: bool = True, use_dropout: bool = True, 
                                dprob: float = 0.3, **kwargs):
    modules= []
    if use_batchnorm: modules.append(nn.BatchNorm1d(in_features))
    if use_dropout: modules.append(nn.Dropout(p=dprob))
    modules.append(nn.Linear(in_features, n_classes))
    
    return nn.Sequential(*modules)
        