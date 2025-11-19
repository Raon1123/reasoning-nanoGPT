"""
This code blocks for common components about nn.Module layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MLPConfig:
    n_embd: int
    bias: bool = False
    dropout: float = 0.0
    activation_fn: callable = None  # default to relu_squared if None

# relu squared activation function
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class MLP(nn.Module):
    def __init__(self, 
                 config: MLPConfig):
        # Note that the original NanoChat used bias=False for linear layers
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.activation_fn = config.activation_fn or relu_squared # Default activation is relu squared
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation_fn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    

    