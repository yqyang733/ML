import sys
from collections import defaultdict
import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import torch

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
MAX_ATOM = 400
MAX_BOND = MAX_ATOM * 2
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
MAX_SEQ_PROTEIN = 1000

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def smiles2mpnnfeature(smiles):
    try: 
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms, fbonds = [], [padding] 
        in_bonds,all_bonds = [], [(-1,-1)] 
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            #print(atom.GetSymbol())
            fatoms.append(atom_features(atom))
            in_bonds.append([])
        #print("fatoms: ",fatoms)

        for bond in mol.GetBonds():
            #print(bond)
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() 
            y = a2.GetIdx()
            #print(x,y)

            b = len(all_bonds)
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0) 
        fbonds = torch.stack(fbonds, 0) 
        agraph = torch.zeros(n_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
        for a in range(n_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2
        # print("fatoms: ",fatoms)
        # print("fbonds: ",fbonds)
        # print("agraph: ",agraph)
        # print("bgraph: ",bgraph)

    except: 
        print('Molecules not found and change to zero vectors..')
        fatoms = torch.zeros(0,39)
        fbonds = torch.zeros(0,50)
        agraph = torch.zeros(0,6)
        bgraph = torch.zeros(0,6)
    #fatoms, fbonds, agraph, bgraph = [], [], [], [] 
    #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
    Natom, Nbond = fatoms.shape[0], fbonds.shape[0]


    ''' 
    ## completion to make feature size equal. 
    MAX_ATOM = 100
    MAX_BOND = 200
    '''
    atoms_completion_num = MAX_ATOM - fatoms.shape[0]
    bonds_completion_num = MAX_BOND - fbonds.shape[0]
    try:
        assert atoms_completion_num >= 0 and bonds_completion_num >= 0
    except:
        raise Exception("Please increasing MAX_ATOM in line 26 utils.py, for example, MAX_ATOM=600 and reinstall it via 'python setup.py install'. The current setting is for small molecule. ")


    fatoms_dim = fatoms.shape[1]
    fbonds_dim = fbonds.shape[1]
    fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
    fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
    agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, MAX_NB)], 0)
    bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, MAX_NB)], 0)
    # print("atom size", fatoms.shape[0], agraph.shape[0])
    # print("bond size", fbonds.shape[0], bgraph.shape[0])
    shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]

def smiles_embed(smiles):
    with open(smiles) as f:
        f1 = f.readlines()
    dict_icv_drug_encoding = {}
    for line in f1:
        dict_icv_drug_encoding[line.split(",")[0]] = smiles2mpnnfeature(line.split(",")[1].strip())
        #break
    #print(dict_icv_drug_encoding)
    return dict_icv_drug_encoding

def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
    else:
        temp = temp [:MAX_SEQ_PROTEIN]
    return temp

def protein_embed(protein):
    with open(protein) as f:
        f1 = f.readlines()
    dict_target_protein_encoding = {}
    for line in f1:
        dict_target_protein_encoding[line.split(",")[0]] = trans_protein(line.split(",")[1].strip())
    # print(len(dict_target_protein_encoding["P0C6X7_3C"]))
    # print(len(f1[0]))
    return dict_target_protein_encoding

def main():
    #smiles = str(sys.argv[1])
    #protein = str(sys.argv[2])
    protein = str(sys.argv[1])
    #DTI = str(sys.argv[3])
    #smiles_embed(smiles)
    protein_embed(protein)

if __name__=="__main__":
    main() 