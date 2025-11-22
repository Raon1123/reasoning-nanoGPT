import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .nanogpt import GPT, GPTConfig

class HRMConfig(GPTConfig):
    """
    Configuration for Hierarchical Reasoning Model.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize HRMConfig.

        Args:
            **kwargs: Keyword arguments for GPTConfig and HRM specific args.
        """
        super().__init__(**kwargs)
        self.n_planner_layers: int = kwargs.get('n_planner_layers', 2)
        self.n_actor_layers: int = kwargs.get('n_actor_layers', 4)
        # Dimension of planner output if different from n_embd
        # For simplicity, we assume same n_embd for now, or project
        self.planner_n_embd: int = kwargs.get('n_embd', 768) 

class HRM(nn.Module):
    """
    Hierarchical Reasoning Model (HRM).
    Consists of a High-Level Planner and a Low-Level Actor (Computer).
    """
    def __init__(self, config: HRMConfig) -> None:
        """
        Initialize HRM.

        Args:
            config (HRMConfig): Configuration object.
        """
        super().__init__()
        self.config = config
        
        # Planner Config
        # We clone the config to avoid side effects
        self.planner_config = GPTConfig(**config.__dict__)
        self.planner_config.n_layer = getattr(config, 'n_planner_layers', 2)
        # Planner might be smaller/lighter
        self.planner = GPT(self.planner_config)
        
        # Actor Config
        self.actor_config = GPTConfig(**config.__dict__)
        self.actor_config.n_layer = getattr(config, 'n_actor_layers', 4)
        self.actor = GPT(self.actor_config)
        
        # Projection from Planner to Actor
        # If n_embd is same, this is just a linear transform to mix features
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
    def forward(self, idx: Optional[torch.Tensor], targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of HRM.

        Args:
            idx (Optional[torch.Tensor]): Input token indices.
            targets (Optional[torch.Tensor]): Target token indices.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Logits, loss, and reward.
        """
        # 1. Planner Pass
        # Extract features from planner (high-level reasoning)
        # We don't necessarily need targets for planner in this simple version,
        # but in full HRM, planner might have its own targets (reasoning steps).
        # Here we use the same input for both.
        planner_features = self.planner.extract_features(idx) # (B, T, C)
        
        # 2. Project Planner Features
        # Condition Actor on Planner features
        # Additive conditioning: Actor Input = Actor Embeddings + Projected Planner Features
        planner_cond = self.proj(planner_features)
        
        # 3. Actor Pass
        # We need to get Actor's own embeddings first
        # GPT.forward with input_embeddings argument
        # But we want to combine Actor's token embeddings with Planner condition
        
        # Get Actor's token embeddings manually to add conditioning
        actor_tok_emb = self.actor.transformer.wte(idx)
        
        # Combine
        combined_embeddings = actor_tok_emb + planner_cond
        
        # Forward Actor with combined embeddings
        # We pass targets to Actor to compute loss
        logits, loss, reward = self.actor(idx=None, targets=targets, input_embeddings=combined_embeddings)
        
        # We return the Actor's outputs as the model outputs
        # In a more complex setup, we might return Planner outputs too or a combined loss
        return logits, loss, reward

    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs: Any) -> torch.Tensor:
        """
        Generate tokens using HRM.

        Args:
            idx (torch.Tensor): Initial token indices.
            max_new_tokens (int): Number of tokens to generate.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Generated token indices.
        """
        # For generation, we need to maintain the hierarchical state
        # This is tricky with the simple additive structure if we just call actor.generate
        # because actor.generate calls actor.forward which needs embeddings.
        
        # We implement a simple generation loop here that mirrors GPT.generate but with HRM logic
        
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward HRM
            logits, _, _ = self(idx_cond)
            
            # Standard generation logic (greedy/sampling)
            # ... (simplified for brevity, just greedy or simple sampling)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
