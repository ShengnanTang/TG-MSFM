import torch

for i in range(torch.cuda.device_count()):
    try:
        x = torch.rand(10, 10).to(f"cuda:{i}")
        y = x @ x
        print(f"GPU {i} OK, result {y[0,0].item()}")
    except Exception as e:
        print(f"GPU {i} BAD: {e}")
