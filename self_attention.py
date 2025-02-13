import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(SelfAttention,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.query = nn.Linear(input_size,hidden_size)
        self.key = nn.Linear(input_size,hidden_size)
        self.value = nn.Linear(input_size,hidden_size)

    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.bmm(q,k.transpose(1,2)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(attn_weights,dim=-1)

        attn_output = torch.bmm(attn_weights,v)
        return attn_output