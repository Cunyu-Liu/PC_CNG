import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ..phase2_pretrain.model import BaseReactionEncoder
from ..phase2_pretrain.data import ReactionDataset
from tqdm import tqdm
import argparse
import os
from typing import Tuple, Dict

class RefinementTrainer:
    """Trainer for iterative hard negative mining"""
    
    def __init__(self, 
                 model: BaseReactionEncoder,
                 temperature: float = 0.1,
                 mining_ratio: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.mining_ratio = mining_ratio
        
    def mine_hard_negatives(self, 
                          negative_pool: DataLoader,
                          device: torch.device) -> DataLoader:
        """Mine hard negatives from the pool"""
        self.model.eval()
        scores = []
        all_negatives = []
        
        with torch.no_grad():
            for batch in tqdm(negative_pool, desc="Mining negatives"):
                reactants = batch['reactants'].to(device)
                products = batch['products'].to(device)
                
                # Get similarity scores
                z_r = self.model(reactants, None, mode="reactants")
                z_p = self.model(None, products, mode="products")
                sim = torch.cosine_similarity(z_r, z_p, dim=-1)
                
                scores.extend(sim.cpu().tolist())
                all_negatives.extend(zip(
                    batch['reactants'],
                    batch['products']
                ))
        
        # Select top k% most similar (hardest negatives)
        k = int(len(scores) * self.mining_ratio)
        indices = torch.topk(torch.tensor(scores), k).indices
        hard_negatives = [all_negatives[i] for i in indices]
        
        # Create new DataLoader with hard negatives
        return self._create_dataloader(hard_negatives)
    
    def train_step(self,
                  pos_loader: DataLoader,
                  neg_loader: DataLoader,
                  optimizer: optim.Optimizer,
                  device: torch.device) -> Tuple[float, float]:
        """Single training step with temperature scaling"""
        self.model.train()
        total_loss = 0.0
        total_pos_sim = 0.0
        
        for pos_batch, neg_batch in zip(pos_loader, neg_loader):
            # Move data to device
            pos_reactants = pos_batch['reactants'].to(device)
            pos_products = pos_batch['products'].to(device)
            neg_reactants = neg_batch['reactants'].to(device)
            neg_products = neg_batch['products'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with temperature scaling
            loss, pos_sim = self.model.get_loss(
                pos_reactants, pos_products,
                neg_reactants, neg_products,
                temperature=self.temperature
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_pos_sim += pos_sim.item()
        
        return total_loss / len(pos_loader), total_pos_sim / len(pos_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--negative_pool', type=str, required=True)
    parser.add_argument('--positive_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--mining_ratio', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = torch.load(args.model_path).to(device)
    
    # Create datasets
    pos_dataset = ReactionDataset(args.positive_data)
    neg_dataset = ReactionDataset(args.negative_pool)
    
    # Create trainer
    trainer = RefinementTrainer(model, 
                              temperature=args.temperature,
                              mining_ratio=args.mining_ratio)
    
    # Training loop
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Mine hard negatives
        neg_loader = DataLoader(neg_dataset, batch_size=args.batch_size)
        hard_neg_loader = trainer.mine_hard_negatives(neg_loader, device)
        
        # Train with hard negatives
        pos_loader = DataLoader(pos_dataset, batch_size=args.batch_size, shuffle=True)
        loss, pos_sim = trainer.train_step(pos_loader, hard_neg_loader, optimizer, device)
        
        print(f"Loss: {loss:.4f} | Pos Sim: {pos_sim:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, f"model_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    main()
