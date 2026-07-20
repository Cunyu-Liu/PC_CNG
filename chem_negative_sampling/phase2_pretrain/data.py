from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict
import pandas as pd
from collections import defaultdict
import numpy as np

class ReactionDataset:
    """Dataset for chemical reaction data"""
    
    def __init__(self, 
                 reaction_csv: str,
                 vocab_file: str = None,
                 max_length: int = 128):
        self.df = pd.read_csv(reaction_csv)
        self.max_length = max_length
        
        # Build vocabulary if not provided
        if vocab_file:
            self.vocab = self._load_vocab(vocab_file)
        else:
            self.vocab = self._build_vocab()
            
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from SMILES strings"""
        vocab = defaultdict(int)
        for rxn in self.df['reaction_smiles']:
            for c in rxn:
                vocab[c] += 1
        # Sort by frequency and assign indices
        sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
        return {c: i+1 for i, (c, _) in enumerate(sorted_vocab)}  # 0 is for padding
        
    def _load_vocab(self, path: str) -> Dict[str, int]:
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            return {line.strip(): i+1 for i, line in enumerate(f)}
            
    def tokenize_smiles(self, smiles: str) -> torch.Tensor:
        """Convert SMILES string to token indices"""
        tokens = [self.vocab.get(c, 0) for c in smiles[:self.max_length]]
        # Pad to max_length
        tokens += [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
        
    def smiles_to_graph(self, smiles: str) -> Data:
        """Convert SMILES string to molecular graph"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic()
            ]
            atom_features.append(features)
            
        # Get bond features and edge indices
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))  # Undirected graph
            
            features = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            edge_features.append(features)
            edge_features.append(features)
            
        return Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float)
        )
        
    def split_reaction(self, rxn_smiles: str) -> Tuple[str, str]:
        """Split reaction SMILES into reactants and products"""
        reactants, products = rxn_smiles.split(">>")
        return reactants, products
        
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Implement based on model type (graph or sequence)
        raise NotImplementedError
