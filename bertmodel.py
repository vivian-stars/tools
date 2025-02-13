import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
 
# 定义一个简化的自注意力层
class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        Q = self.query(x)  # (batch_size, n, hidden_dim)
        K = self.key(x)    # (batch_size, n, hidden_dim)
        V = self.value(x)  # (batch_size, n, hidden_dim)
        
        Q = Q.permute(0, 2, 1)  # (batch_size, hidden_dim, n)
        K = K.permute(0, 2, 1)  # (batch_size, hidden_dim, n)
        
        energy = torch.bmm(Q, K)  # (batch_size, n, n)
        attention = torch.softmax(energy, dim=2)  # (batch_size, n, n)
        
        V = V.permute(0, 2, 1)  # (batch_size, n, hidden_dim)
        context = torch.bmm(attention, V)  # (batch_size, n, hidden_dim)
        
        return context.permute(0, 2, 1)  # (batch_size, hidden_dim, n)
 
# 使用自注意力层的Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SimpleSelfAttention(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # 自注意力
        attn_output = self.attention(x)
        attn_output = self.layer_norm1(x + attn_output)
        
        # 前馈神经网络
        ffn_output = self.feed_forward(attn_output)
        ffn_output = self.layer_norm2(attn_output + ffn_output)
        
        return ffn_output
 
# 实例化Transformer块
hidden_dim = 512
transformer_block = TransformerBlock(hidden_dim)
 
# 输入数据
input_data = torch.randn(10, 3, 512)  # (batch_size, sequence_length, hidden_dim)
 
# 通过Transformer块
output = transformer_block(input_data)
print(output.shape)