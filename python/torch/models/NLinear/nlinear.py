import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        #if td:
        #    self.Linear = TimeDistributed(self.Linear)
        
        # Use this line if you want to visualize the weights
        self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self._reinitialize()
    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'Linear' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]