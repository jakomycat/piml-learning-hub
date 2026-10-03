import torch
import torch.nn as nn

class PINNSolution(nn.Module):
    """Multi Layer Perceptro for aproximate u(t, x).

    Input: Tensor with dimension (N, 2), where each row is (t, x).
    Output: Tensor with dimension (N, 1), where the row is u(t, x).
    """
    def __init__(self, layer_sizes):
        super(PINNSolution, self).__init__()
        
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
                
        self.network = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, t, x):
        """Forward pass to evaluate u(t, x)."""
        inputs = torch.cat([t, x], dim=1) # dim: (N, 2)
        
        return self.network(inputs)
    

class PINNDiscovery(nn.Module):
    """
    Multi Layer Perceptro for discover u(t, x) parameters.

    These are the same inputs and outputs as PINNSolution.
    """
    def __init__(self, layer_sizes):
        super(PINNDiscovery, self).__init__()
        
        layers = []
                
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
                
        self.network = nn.Sequential(*layers)
        self._init_weights()
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0]))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0]))
        
    def _init_weights(self):
        """Xavier initialization."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, t, x):
        """Forward pass to evaluate u(t, x)."""
        inputs = torch.cat([t, x], dim=1)
        
        return self.network(inputs)