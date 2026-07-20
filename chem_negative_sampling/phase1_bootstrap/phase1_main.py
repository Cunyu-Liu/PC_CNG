import pandas as pd
from rule_based_generation import generate_incompatible_pairs, save_negative_samples as save_rule_based
from heuristic_combinations import create_invalid_combinations, save_negative_samples as save_heuristic
from low_yield_processor import (load_reaction_data_with_yields, 
                                filter_low_yield_reactions,
                                process_low_yield_reactions,
                                save_negative_samples as save_low_yield)
from typing import List
import os

def combine_negative_samples(output_dir: str = "../data") -> pd.DataFrame:
    """Combine all negative samples from different methods"""
    files = [
        os.path.join(output_dir, "rule_based_negatives.csv"),
        os.path.join(output_dir, "heuristic_negatives.csv"), 
        os.path.join(output_dir, "low_yield_negatives.csv")
    ]
    
    dfs = []
    for f in files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    
    if not dfs:
        return pd.DataFrame(columns=["reaction_smiles"])
    
    combined = pd.concat(dfs).drop_duplicates()
    return combined

def run_phase1(reaction_data_path: str = None, 
               yield_data_path: str = None,
               output_dir: str = "../data",
               n_heuristic_samples: int = 1000,
               max_yield: float = 10.0):
    """Run all Phase 1 bootstrapping components"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Running Rule-Based Generation ===")
    rule_samples = generate_incompatible_pairs()
    save_rule_based(rule_samples, os.path.join(output_dir, "rule_based_negatives.csv"))
    
    print("\n=== Running Heuristic Combinations ===")
    if reaction_data_path:
        df = pd.read_csv(reaction_data_path)
        heuristic_samples = create_invalid_combinations(df, n_samples=n_heuristic_samples)
        save_heuristic(heuristic_samples, os.path.join(output_dir, "heuristic_negatives.csv"))
    
    print("\n=== Processing Low-Yield Reactions ===")
    if yield_data_path:
        df = load_reaction_data_with_yields(yield_data_path)
        low_yield_df = filter_low_yield_reactions(df, max_yield=max_yield)
        low_yield_samples = process_low_yield_reactions(low_yield_df)
        save_low_yield(low_yield_samples, os.path.join(output_dir, "low_yield_negatives.csv"))
    
    print("\n=== Combining All Negative Samples ===")
    combined = combine_negative_samples(output_dir)
    combined.to_csv(os.path.join(output_dir, "phase1_combined_negatives.csv"), index=False)
    print(f"Generated {len(combined)} total negative samples")

if __name__ == "__main__":
    # Example usage (would need real data paths)
    run_phase1(
        # reaction_data_path="../data/uspto_reactions.csv",  # Uncomment with real data
        # yield_data_path="../data/reactions_with_yields.csv"  # Uncomment with real data
    )
