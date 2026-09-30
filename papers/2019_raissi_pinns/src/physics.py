import torch

def compute_pde_residual(model, t, x, viscosity):
    """Calculate the residual f(t, x) of Burgers Equation 1D."""
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    # Get u(t, x)
    u = model(t, x)
    
    # Get partial derivates
    u_g = torch.autograd.grad(
        outputs=u, inputs=[t, x], grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )
    
    u_t = u_g[0]
    u_x = u_g[1]
    
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    # Residual f(t, x)
    f = u_t + u * u_x - viscosity * u_xx
    
    return f