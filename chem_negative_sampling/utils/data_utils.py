import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import json
import os
from sklearn.model_selection import train_test_split

def load_reaction_data(file_path: str) -> pd.DataFrame:
    """Load reaction data from CSV file"""
    df = pd.read_csv(file_path)
    
    # Basic validation
    required_cols = {'reaction_smiles', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")
        
    return df

def save_reaction_data(df: pd.DataFrame, file_path: str):
    """Save reaction data to CSV file"""
    required_cols = {'reaction_smiles', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Data must contain columns: {required_cols}")
        
    df.to_csv(file_path, index=False)

def split_dataset(df: pd.DataFrame, 
                 test_size: float = 0.2,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/test sets with stratified sampling"""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    return train_df, test_df

def batch_process_data(df: pd.DataFrame, 
                      batch_size: int,
                      process_fn: callable) -> List[Dict]:
    """Process data in batches"""
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    return results

def save_metrics(metrics: Dict, file_path: str):
    """Save evaluation metrics to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(file_path: str) -> Dict:
    """Load evaluation metrics from JSON file"""
    with open(file_path) as f:
        return json.load(f)

def create_dataset_stats(df: pd.DataFrame) -> Dict:
    """Calculate basic dataset statistics"""
    stats = {
        'total_samples': len(df),
        'positive_samples': int(df['label'].sum()),
        'negative_samples': len(df) - int(df['label'].sum()),
        'class_ratio': float(df['label'].mean())
    }
    
    # Add reaction length stats if smiles column exists
    if 'reaction_smiles' in df.columns:
        lengths = df['reaction_smiles'].str.len()
        stats.update({
            'avg_smiles_length': float(lengths.mean()),
            'max_smiles_length': int(lengths.max()),
            'min_smiles_length': int(lengths.min())
        })
        
    return stats
