import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, absolute_positional_bias=None, relative_positional_bias=None, return_attention=False):
        x = x.permute(1,0,-1)
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        d_k = q.size()[-1]

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        
        scaling_factor = 2
        attn_logits = attn_logits / math.sqrt(scaling_factor * d_k)
        
        # add relative positional bias
        if relative_positional_bias is not None:
            attn_logits += relative_positional_bias
        # add absolute positional bias
        if absolute_positional_bias is not None:
            attn_logits = absolute_positional_bias
            
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o