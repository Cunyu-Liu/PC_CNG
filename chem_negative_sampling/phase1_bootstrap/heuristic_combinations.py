import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from typing import List, Tuple
from tqdm import tqdm

def load_reaction_data(path: str) -> pd.DataFrame:
    """Load reaction data from CSV file"""
    return pd.read_csv(path)

def create_invalid_combinations(df: pd.DataFrame, n_samples: int = 1000) -> List[str]:
    """Create invalid reaction combinations from valid reactions"""
    negative_samples = []
    reactions = df['reaction_smiles'].tolist()
    
    for _ in tqdm(range(n_samples), desc="Generating negative samples"):
        # Randomly select two different reactions
        rxn1, rxn2 = random.sample(reactions, 2)
        
        # Split into reactants and products
        reactants1, products1 = rxn1.split(">>")
        reactants2, products2 = rxn2.split(">>")
        
        # Create invalid combination: reactants from rxn1 + products from rxn2
        invalid_rxn = f"{reactants1}>>{products2}"
        
        # Verify atom conservation (should fail for negative samples)
        try:
            rxn = AllChem.ReactionFromSmarts(invalid_rxn)
            if rxn.Validate()[1]:  # Check if reaction is valid
                continue  # Skip actually valid combinations
        except:
            pass  # Invalid reactions are what we want
            
        negative_samples.append(invalid_rxn)
    
    return negative_samples

def save_negative_samples(samples: List[str], output_path: str):
    """Save negative samples to file"""
    df = pd.DataFrame({"reaction_smiles": samples})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage (would need real USPTO/Reaxys data)
    print("Loading reaction data...")
    # df = load_reaction_data("../data/uspto_reactions.csv")  # Uncomment with real data
    df = pd.DataFrame({"reaction_smiles": ["CC(=O)O.O>>CC(=O)O", "CN>>CN"]})  # Example data
    
    print("Generating heuristic combinations...")
    samples = create_invalid_combinations(df, n_samples=100)
    
    print(f"Generated {len(samples)} heuristic negative samples")
    save_negative_samples(samples, "../data/heuristic_negatives.csv")
