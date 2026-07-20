from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Optional
import numpy as np

def validate_reaction_smiles(rxn_smiles: str) -> bool:
    """Validate that a reaction SMILES string is properly formatted"""
    try:
        reactants, products = rxn_smiles.split(">>")
        if not reactants or not products:
            return False
            
        # Check if molecules can be parsed
        for smi in reactants.split("."):
            if not Chem.MolFromSmiles(smi):
                return False
        for smi in products.split("."):
            if not Chem.MolFromSmiles(smi):
                return False
                
        return True
    except:
        return False

def calculate_reaction_properties(rxn_smiles: str) -> Optional[Dict]:
    """Calculate properties for a reaction SMILES string"""
    if not validate_reaction_smiles(rxn_smiles):
        return None
        
    reactants, products = rxn_smiles.split(">>")
    reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
    product_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".")]
    
    # Calculate properties for reactants and products
    def get_avg_props(mols):
        props = []
        for mol in mols:
            if mol:
                props.append({
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'RingCount': Descriptors.RingCount(mol)
                })
        return {k: np.mean([p[k] for p in props]) for k in props[0]} if props else None
    
    reactant_props = get_avg_props(reactant_mols)
    product_props = get_avg_props(product_mols)
    
    if not reactant_props or not product_props:
        return None
        
    # Calculate reaction-level properties
    return {
        'reactants': reactant_props,
        'products': product_props,
        'delta_MW': product_props['MW'] - reactant_props['MW'],
        'delta_LogP': product_props['LogP'] - reactant_props['LogP']
    }

def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

def reaction_fingerprint(rxn_smiles: str, radius=2, nBits=2048) -> Optional[np.ndarray]:
    """Generate reaction fingerprint"""
    if not validate_reaction_smiles(rxn_smiles):
        return None
        
    reactants, products = rxn_smiles.split(">>")
    
    def get_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) if mol else None
        
    # Get fingerprints for all components
    reactant_fps = [get_fp(smi) for smi in reactants.split(".")]
    product_fps = [get_fp(smi) for smi in products.split(".")]
    
    # Combine fingerprints
    if all(reactant_fps) and all(product_fps):
        rxn_fp = np.zeros(nBits, dtype=int)
        for fp in reactant_fps + product_fps:
            rxn_fp |= np.array(fp)
        return rxn_fp
    return None
