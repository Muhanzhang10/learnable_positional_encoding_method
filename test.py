import torch

from transformer import TransformerSentenceEncoder

model = TransformerSentenceEncoder(30)
res = model(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]))

print(res.shape)