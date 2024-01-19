import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from multihead_attention import MultiheadAttention




"""_summary_ This is from T5
This function embeds relative positions
    Translate relative position to a bucket number for relative attention. We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.
"""
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class TransformerSentenceEncoder(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        num_encoder_layers=1,
        max_rel_pos=128,  # max relative position distance
        max_seq_len=256,
        num_attention_heads = 8,
        embedding_dim=768,
        rel_pos_bins=32  # num positional buckets
        ):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.max_rel_pos = max_rel_pos
        self.embedding_dim = embedding_dim
        self.seq_len = max_seq_len
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.attn_scale_factor = 2
        self.num_attention_heads = num_attention_heads
        self.pos = nn.Embedding(self.seq_len + 1, self.embedding_dim)  # absolute positional embedding
        self.pos_q_linear = nn.Linear(self.embedding_dim, self.embedding_dim)  #U_Q d*d projection matrice for the positional embedding
        self.pos_k_linear = nn.Linear(self.embedding_dim, self.embedding_dim)  #U_K
        self.pos_scaling = float(self.embedding_dim / num_attention_heads * self.attn_scale_factor) ** -0.5 
        self.pos_ln = nn.LayerNorm(self.embedding_dim)
        
        self.layers = nn.ModuleList(
            [
                MultiheadAttention(
                    input_dim=self.embedding_dim,
                    embed_dim=self.embedding_dim,
                    num_heads=num_attention_heads,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        #Initialize relative position
        self.rel_pos_bins = rel_pos_bins
        self.relative_attention_bias = nn.Embedding(self.rel_pos_bins + 1, self.num_attention_heads)
        context_position = torch.arange(self.seq_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(self.seq_len, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        self.rp_bucket = relative_position_bucket(
                relative_position,
                num_buckets=self.rel_pos_bins,
                max_distance=self.max_rel_pos
        )
        
    
    def get_rel_pos_bias(self, x):
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()
        
        
    def forward(
        self,
        tokens
    ):

        # Compute relative positional bias from T5
        rel_pos_bias = self.get_rel_pos_bias(tokens)
        # Embed tokens
        x = self.embed_tokens(tokens)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        seq_len = x.size(0)
        # compute absolute positional bias
        weight = self.pos_ln(self.pos.weight[:seq_len, :])  # Absolute positional embedding: pi & pj, pos_ln used to normalize parameters
        pos_q =  self.pos_q_linear(weight).view(seq_len, self.num_attention_heads, -1).transpose(0, 1) * self.pos_scaling
        pos_k =  self.pos_k_linear(weight).view(seq_len, self.num_attention_heads, -1).transpose(0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))

        # expand encoding to batch size
        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1), -1, -1, -1)
        rel_pos_bias = rel_pos_bias.unsqueeze(0).expand(x.size(1), -1, -1, -1)


        for layer in self.layers:
            x = layer(x, absolute_positional_bias=abs_pos_bias, relative_positional_bias=rel_pos_bias)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        return sentence_rep
