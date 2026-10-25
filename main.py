import torch

tensor = torch.ones(5)
# Movendo o tensor para o acelerador se ele existir
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

