import torch
import torch.nn as nn

class DeepONet(nn.Module):
    """"""
    def __init__(self, m, p, hidden):
        super().__init__()
        
        # Branch net
        self.branch_net = nn.Sequential(
            nn.Linear(m, hidden),
            nn.Tanh(),
            
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            
            nn.Linear(hidden, p)
        )
        
        # Trunk net
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),

            nn.Linear(hidden, p)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()
        
    def _init_weights(self):
        """Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, u0_sensors, xt):
        """
        Args:
            u0_sensors: dim (N, m).
            xt: dim (Q, 2).
        """
        branch = self.branch_net(u0_sensors)
        trunk = self.trunk_net(xt)
        
        out = torch.einsum("np,qp->nq", branch, trunk)
        out = out + self.bias
        
        return out
        