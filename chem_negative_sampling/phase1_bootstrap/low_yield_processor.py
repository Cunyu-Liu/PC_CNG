import pandas as pd
from rdkit import Chem
from typing import List

def load_reaction_data_with_yields(path: str) -> pd.DataFrame:
    """Load reaction data with yield information"""
    return pd.read_csv(path)

def filter_low_yield_reactions(df: pd.DataFrame, max_yield: float = 10.0) -> pd.DataFrame:
    """Filter reactions with yields below threshold"""
    return df[df['yield'] < max_yield]

def process_low_yield_reactions(df: pd.DataFrame) -> List[str]:
    """Process low-yield reactions into negative samples"""
    return df['reaction_smiles'].tolist()

def save_negative_samples(samples: List[str], output_path: str):
    """Save negative samples to file"""
    df = pd.DataFrame({"reaction_smiles": samples})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage (would need real data with yields)
    print("Loading reaction data with yields...")
    # df = load_reaction_data_with_yields("../data/reactions_with_yields.csv")  # Uncomment with real data
    df = pd.DataFrame({
        "reaction_smiles": ["CC(=O)O.O>>CC(=O)O", "CN>>CN"],
        "yield": [85.0, 3.5]  # Example data
    })
    
    print("Filtering low-yield reactions...")
    low_yield_df = filter_low_yield_reactions(df, max_yield=10.0)
    
    print("Processing low-yield reactions...")
    samples = process_low_yield_reactions(low_yield_df)
    
    print(f"Found {len(samples)} low-yield negative samples")
    save_negative_samples(samples, "../data/low_yield_negatives.csv")
