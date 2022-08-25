import torch

a = torch.tensor([[1, 2, 3], [1, 2, 3]])
mask = torch.tensor([[0, 1, 1]])
print(a*mask)