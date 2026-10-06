import torch

def generate_data(N, K, m, nx, nt, alpha, T, seed):
    g = torch.Generator().manual_seed(seed)
    
    # Initial conditions in the sensors
    k = torch.arange(1, K + 1, dtype=torch.float32)
    a = torch.randn(N, K, generator=g) / k[None, :]
    
    x_s = torch.linspace(0, 1, m) # Sensors x_1, x_2, ..., x_m
    S = torch.sin(torch.pi * k[:, None] * x_s[None, :])
    u0_sensors = a @ S
    
    # Grid
    x_grid = torch.linspace(0, 1, nx)
    t_grid = torch.linspace(0, T, nt)
    X, Tt = torch.meshgrid(x_grid, t_grid, indexing="ij")
    xt = torch.stack([X, Tt], dim=-1).reshape(-1, 2)
    
    # Exact solution
    xq, tq = xt[:, 0], xt[:, 1]
    decay = torch.exp(-alpha * (torch.pi * k[:, None]) ** 2 * tq[None, :])
    Phi = decay * torch.sin(torch.pi * k[:, None] * xq[None, :])
    target = a @ Phi
    
    return u0_sensors, xt, target