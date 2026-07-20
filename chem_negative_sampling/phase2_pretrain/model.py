import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple

class BaseReactionEncoder(nn.Module):
    """Base class for reaction encoders using contrastive learning"""
    
    def __init__(self, 
                 node_dim: int = 64,
                 edge_dim: int = 32,
                 hidden_dim: int = 128,
                 out_dim: int = 256):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def encode_reactants(self, reactants):
        """Encode reactants (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def encode_products(self, products):
        """Encode products (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def forward(self, reactants, products, mode: str = "reactants") -> torch.Tensor:
        """Forward pass for contrastive learning"""
        if mode == "reactants":
            h = self.encode_reactants(reactants)
        elif mode == "products":
            h = self.encode_products(products)
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        # Project embeddings for contrastive learning
        z = self.projection_head(h)
        return z
        
    def get_loss(self, 
                pos_reactants, pos_products,
                neg_reactants, neg_products,
                temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive loss between positive and negative pairs"""
        # Encode positive pairs
        z_pos_r = self(pos_reactants, None, mode="reactants")
        z_pos_p = self(None, pos_products, mode="products")
        
        # Encode negative pairs
        z_neg_r = self(neg_reactants, None, mode="reactants")
        z_neg_p = self(None, neg_products, mode="products")
        
        # Compute contrastive loss
        pos_sim = torch.cosine_similarity(z_pos_r, z_pos_p, dim=-1) / temperature
        neg_sim = torch.cosine_similarity(z_pos_r, z_neg_p, dim=-1) / temperature
        
        logits = torch.cat([pos_sim, neg_sim], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_sim),  # Positive pairs
            torch.zeros_like(neg_sim)   # Negative pairs
        ], dim=0)
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss, pos_sim.mean()
