import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .nanogpt import GPT, GPTConfig

class TRMConfig(GPTConfig):
    """
    Configuration for Tiny Recursive Model.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize TRMConfig.

        Args:
            **kwargs: Keyword arguments.
        """
        self.n_recurrence: int = kwargs.pop('n_recurrence', 1) # Number of recursive steps
        super().__init__(**kwargs)
        self.use_reward_head: bool = True

class TRM(GPT):
    """
    Tiny Recursive Model (TRM).
    Extends GPT to support recursive latent state updates and token-level rewards.
    """
    def __init__(self, config: TRMConfig) -> None:
        """
        Initialize TRM.

        Args:
            config (TRMConfig): Configuration object.
        """
        # Ensure reward head is enabled
        config.use_reward_head = True
        super().__init__(config)
        self.n_recurrence: int = getattr(config, 'n_recurrence', 1)

    def forward(self, idx: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, input_embeddings: Optional[torch.Tensor] = None) -> Any:
        """
        Forward pass with recurrence.
        
        Args:
            idx (Optional[torch.Tensor]): Input token indices.
            targets (Optional[torch.Tensor]): Target token indices.
            input_embeddings (Optional[torch.Tensor]): Optional input embeddings (overrides idx).

        Returns:
            Any: Model outputs (logits, loss, reward).
        """
        device = idx.device if idx is not None else input_embeddings.device
        
        if input_embeddings is not None:
            b, t, _ = input_embeddings.size()
            tok_emb = input_embeddings
        else:
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        
        # Initial embedding
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Recursive application of transformer blocks
        # We apply the entire stack of layers n_recurrence times
        for r in range(self.n_recurrence):
            for block in self.transformer.h:
                x = block(x)
            # Optional: Add some mixing or normalization between recurrences?
            # For now, simple pass-through.
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        
        reward = None
        if self.reward_head is not None:
            reward = self.reward_head(x)
            
        return logits, loss, reward

    def step_recursive(self, idx: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> None:
        """
        Perform one step of recursive update.
        Placeholder for specific recursive logic.

        Args:
            idx (torch.Tensor): Input indices.
            hidden_state (Optional[torch.Tensor]): Hidden state.
        """
        # TODO: Implement specific recursive logic if defined.
        pass


