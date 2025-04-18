#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:20
# @Author  : Li Xiao
# @File    : gcn_model.py
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d, LeakyReLU

class GAT(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, heads=8):
        super(GAT, self).__init__()
        # Đảm bảo heads là số nguyên khi chia
        self.conv1 = GATConv(n_in, n_hid, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm1d(n_hid * heads)
        
        heads2 = max(1, heads // 2)  # Đảm bảo ít nhất là 1
        self.conv2 = GATConv(n_hid * heads, n_hid, heads=heads2, dropout=dropout)
        self.bn2 = BatchNorm1d(n_hid * heads2)
        
        heads3 = max(1, heads // 4)  # Đảm bảo ít nhất là 1
        self.conv3 = GATConv(n_hid * heads2, n_hid, heads=heads3, dropout=dropout)
        self.bn3 = BatchNorm1d(n_hid * heads3)
        
        # Layer cuối cùng luôn có 1 head và output là số classes
        self.conv4 = GATConv(n_hid * heads3, n_out, heads=1, dropout=dropout)
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)