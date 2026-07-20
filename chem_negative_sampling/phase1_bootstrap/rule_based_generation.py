import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from typing import List

def generate_incompatible_pairs() -> List[str]:
    """Generate SMILES pairs with known incompatible functional groups"""
    # Define incompatible functional group combinations
    incompatible_pairs = [
        # Acid + Base
        ("[CX3](=O)[OX2H1]", "[NX3;H2,H1;!$(NC=O)]"),
        # Aldehyde + Amine (forms imine)
        ("[CX3H1](=O)[#6]", "[NX3;H2,H1;!$(NC=O)]"),
        # Alcohol + Carboxylic acid (forms ester)
        ("[CX3](=O)[OX2H1]", "[OX2H]"),
    ]
    
    # Generate example molecules for each pair
    negative_samples = []
    for fg1, fg2 in incompatible_pairs:
        # Create simple molecules with these functional groups
        mol1 = Chem.MolFromSmarts(fg1)
        mol2 = Chem.MolFromSmarts(fg2)
        
        # Combine them into a "reaction" that shouldn't work
        if mol1 and mol2:
            smi1 = Chem.MolToSmiles(mol1)
            smi2 = Chem.MolToSmiles(mol2)
            negative_samples.append(f"{smi1}.{smi2}>>")
    
    return negative_samples

def save_negative_samples(samples: List[str], output_path: str):
    """Save negative samples to file"""
    df = pd.DataFrame({"reaction_smiles": samples})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Generate and save initial negative samples
    samples = generate_incompatible_pairs()
    save_negative_samples(samples, "../data/rule_based_negatives.csv")
    print(f"Generated {len(samples)} rule-based negative samples")
