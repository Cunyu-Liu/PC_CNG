import torch
from torch.utils.data import DataLoader
from ..phase2_pretrain.model import BaseReactionEncoder
from ..phase2_pretrain.data import ReactionDataset
from ..utils.data_utils import load_reaction_data
from .metrics import NegativeSampleEvaluator
import argparse
import json
import os

def evaluate_model(model: BaseReactionEncoder,
                  test_loader: DataLoader,
                  device: torch.device) -> dict:
    """Evaluate model performance on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            reactants = batch['reactants'].to(device)
            products = batch['products'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions
            z_r = model(reactants, None, mode="reactants")
            z_p = model(None, products, mode="products")
            preds = torch.sigmoid(torch.cosine_similarity(z_r, z_p, dim=-1))
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    return {
        'predictions': all_preds,
        'labels': all_labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = torch.load(args.model_path).to(device)
    
    # Load and prepare test data
    test_df = load_reaction_data(args.test_data)
    test_dataset = ReactionDataset(args.test_data)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    evaluator = NegativeSampleEvaluator(
        pos_data_path=args.test_data,
        neg_data_path=args.test_data  # Would use separate files in practice
    )
    metrics = {
        'model_performance': evaluator.evaluate_utility(results['predictions'], results['labels']),
        'dataset_stats': evaluator.create_dataset_stats(test_df)
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()
