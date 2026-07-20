import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .model import BaseReactionEncoder

class TransformerReactionEncoder(BaseReactionEncoder):
    """Transformer-based reaction encoder using SMILES strings"""
    
    def __init__(self,
                 vocab_size: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__(node_dim=d_model, edge_dim=0, hidden_dim=d_model, out_dim=d_model)
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        
    def encode_smiles(self, src, src_mask=None, src_key_padding_mask=None):
        """Encode SMILES sequences"""
        src = self.token_embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer(src, src_mask, src_key_padding_mask)
        return self.proj(output.mean(dim=1))  # Mean pooling over sequence
        
    def encode_reactants(self, reactants):
        """Encode reactants (batch of tokenized SMILES)"""
        return self.encode_smiles(
            reactants.input_ids,
            src_key_padding_mask=reactants.attention_mask
        )
        
    def encode_products(self, products):
        """Encode products (batch of tokenized SMILES)"""
        return self.encode_smiles(
            products.input_ids,
            src_key_padding_mask=products.attention_mask
        )

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
