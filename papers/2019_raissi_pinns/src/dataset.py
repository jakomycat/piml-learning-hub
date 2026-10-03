import torch

def generate_domain_samples(n_u, n_f, x_min, x_max, t_min, t_max, device):
    """Generates training tensors for data IC + BC and residuals."""
    # Initial conditions
    t_ic = torch.full((n_u // 2, 1), t_min, dtype=torch.float32)
    x_ic = torch.empty((n_u // 2, 1), dtype=torch.float32).uniform_(x_min, x_max)
    u_ic = -torch.sin(torch.pi * x_ic)
    
    # Boundary conditions
    t_bc = torch.empty((n_u // 2, 1), dtype=torch.float32).uniform_(t_min, t_max)
    
    x_bc_left = torch.full((n_u // 4, 1), x_min, dtype=torch.float32)
    x_bc_right = torch.full((n_u // 4, 1), x_max, dtype=torch.float32)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    
    u_bc = torch.zeros_like(x_bc)
    
    t_u = torch.cat([t_ic, t_bc], dim=0).to(device)
    x_u = torch.cat([x_ic, x_bc], dim=0).to(device)
    u_u = torch.cat([u_ic, u_bc], dim=0).to(device)
    
    # Collocation points
    t_f = torch.empty((n_f, 1), dtype=torch.float32).uniform_(t_min, t_max).to(device)
    x_f = torch.empty((n_f, 1), dtype=torch.float32).uniform_(x_min, x_max).to(device)
    
    return (t_u, x_u, u_u), (t_f, x_f)


def generate_noisy_data(n_obs, x_min, x_max, t_min, t_max, device, noise_level=0.05):
    """Generates noisy data to try to discover the parameters "lambda".

    The data is generated considering lambda_1=6.0 and lambda_2=1.0.
    """
    x_data = torch.empty((n_obs, 1), dtype=torch.float32).uniform_(x_min, x_max).to(device)
    t_data = torch.empty((n_obs, 1), dtype=torch.float32).uniform_(t_min, t_max).to(device)

    u_exact = 2 * (0.5**2) * (1.0 / torch.cosh(0.5 * x_data - 4 * (0.5**3) * t_data))**2

    noise = noise_level * torch.std(u_exact) * torch.randn_like(u_exact)

    u_obs = u_exact + noise

    return (x_data, t_data), u_obs