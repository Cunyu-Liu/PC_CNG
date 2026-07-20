import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .data import ReactionDataset
from .model import BaseReactionEncoder
from .gnn import GNNReactionEncoder
from .transformer import TransformerReactionEncoder
from tqdm import tqdm
import argparse
import os
from typing import Tuple, Dict

def train_epoch(model: BaseReactionEncoder, 
               train_loader: DataLoader,
               optimizer: optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_pos_sim = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        pos_reactants = batch['pos_reactants'].to(device)
        pos_products = batch['pos_products'].to(device)
        neg_reactants = batch['neg_reactants'].to(device)
        neg_products = batch['neg_products'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss, pos_sim = model.get_loss(
            pos_reactants, pos_products,
            neg_reactants, neg_products
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_pos_sim += pos_sim.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_sim = total_pos_sim / len(train_loader)
    return avg_loss, avg_sim

def evaluate(model: BaseReactionEncoder,
             val_loader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_pos_sim = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pos_reactants = batch['pos_reactants'].to(device)
            pos_products = batch['pos_products'].to(device)
            neg_reactants = batch['neg_reactants'].to(device)
            neg_products = batch['neg_products'].to(device)
            
            loss, pos_sim = model.get_loss(
                pos_reactants, pos_products,
                neg_reactants, neg_products
            )
            
            total_loss += loss.item()
            total_pos_sim += pos_sim.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_sim = total_pos_sim / len(val_loader)
    return avg_loss, avg_sim

def save_checkpoint(model: BaseReactionEncoder,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['gnn', 'transformer'], required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = ReactionDataset(args.train_data)
    val_dataset = ReactionDataset(args.val_data)
    
    # Create model
    if args.model_type == 'gnn':
        model = GNNReactionEncoder().to(device)
    else:
        model = TransformerReactionEncoder().to(device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_sim = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Pos Sim: {train_sim:.4f}")
        
        # Evaluate
        val_loss, val_sim = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Pos Sim: {val_sim:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch,
                {'train_loss': train_loss, 'val_loss': val_loss},
                os.path.join(args.output_dir, 'best_model.pt')
            )

if __name__ == "__main__":
    main()
