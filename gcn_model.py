#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:20
# @Author  : Li Xiao
# @File    : gcn_model.py
from torch import nn
import torch.nn.functional as F
from layer import GraphConvolution
import torch

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5):
        super(GCN, self).__init__()
        # Graph Convolution layers
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.gc3 = GraphConvolution(n_hid, n_hid)  # Additional GCN layer
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid)
        
        # Attention layers
        self.attention1 = AttentionLayer(n_hid)
        self.attention2 = AttentionLayer(n_hid)
        
        # Dropout layers
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
        
        # Final classification layer
        self.fc = nn.Linear(n_hid, n_out)
        
        # Layer for residual connection
        self.residual = nn.Linear(n_in, n_hid) if n_in != n_hid else nn.Identity()
        
        self.dropout = dropout

    def forward(self, input, adj):
        # First GCN layer with residual connection
        identity = self.residual(input)
        x = self.gc1(input, adj)
        x = self.bn1(x)
        x = F.elu(x)
        x = x + identity  # Residual connection
        x = self.attention1(x)  # Apply attention
        x = self.dp1(x)
        
        # Second GCN layer
        identity = x
        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = F.elu(x)
        x = x + identity  # Residual connection
        x = self.attention2(x)  # Apply attention
        x = self.dp2(x)
        
        # Third GCN layer
        identity = x
        x = self.gc3(x, adj)
        x = self.bn3(x)
        x = F.elu(x)
        x = x + identity  # Residual connection
        x = self.dp3(x)
        
        # Final classification
        x = self.fc(x)
        
        return x