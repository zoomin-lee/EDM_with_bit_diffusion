import os
import pickle
import numpy as np
import pandas as pd

import rdkit
import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')


def main():
    data_dir = 'molecule3d/data/data/raw'
    name_list = [
        'combined_mols_0_to_1000000.sdf',
        'combined_mols_1000000_to_2000000.sdf',
        'combined_mols_2000000_to_3000000.sdf',
        'combined_mols_3000000_to_3899647.sdf'
    ]
    
    data_list = [] # mol_id, atom_type, xyz coordinates
    mol_size_list = []
    smiles_list = []
    

    cnt = 0
    for i in range(len(name_list)):
        
        filename = name_list[i]
        filepath = os.path.join(data_dir, filename)
        
        suppl = Chem.SDMolSupplier(filepath)
        
        for mi, mol in enumerate(suppl):
            
            if cnt%500000 == 0:
                print(f"molecule no.{cnt}")
            
            sml = Chem.MolToSmiles(mol)
            mol_size = mol.GetNumAtoms()

            smiles_list.append(sml)
            mol_size_list.append(mol_size)

            for ai, atom in enumerate(mol.GetAtoms()):
                positions = mol.GetConformer().GetAtomPosition(ai)
                data = np.array([cnt, atom.GetAtomicNum(), positions.x, positions.y, positions.z])
                data_list.append(data)
            
            cnt += 1
                
        
    dataset = np.vstack(data_list)
    print(f"data shape: {dataset.shape}")
    np.save('mol_3d_data.npy', dataset)

    mol_size_list = np.asarray(mol_size_list)
    print(f"# of molecules: {mol_size_list.size}")
    np.save('mol_3d_size.npy', mol_size_list)

    with open("mol_3d_smiles.txt", 'w') as f:
        for s in smiles_list:
            f.write(s)
            f.write('\n')
                

if __name__ == "__main__":
    main()