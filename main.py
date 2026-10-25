import torch

tensor = torch.rand(3,4)

print(f"Forma do tensor: {tensor.shape}")
print(f"Tipos do tensor tensor: {tensor.dtype}")
print(f"Local de armazenamento: {tensor.device}")

print(torch.accelerator.is_available())
