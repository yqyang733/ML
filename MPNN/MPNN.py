import sys
import rdkit
from rdkit import Chem, DataStructs

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
MAX_ATOM = 400
MAX_BOND = MAX_ATOM * 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def mpnn_collate_func(x):
    mpnn_feature = [i[0] for i in x]
    mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
    from torch.utils.data.dataloader import default_collate
    x_remain = [list(i[1:]) for i in x]
    x_remain_collated = default_collate(x_remain)
    return [mpnn_feature] + x_remain_collated

def mpnn_feature_collate_func(x):
    #print("x:",x)
    #print("aaaaa")
    N_atoms_scope = torch.cat([i[4] for i in x], 0)
    #print("x[0][0]:",x[0][0])
    f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
    f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
    # f_a = torch.cat([x[j][0] for j in range(len(x))], 0)
    # f_b = torch.cat([x[j][1] for j in range(len(x))], 0)
    agraph_lst, bgraph_lst = [], []
    for j in range(len(x)):
        agraph_lst.append(x[j][2].unsqueeze(0))
        bgraph_lst.append(x[j][3].unsqueeze(0))
        # agraph_lst.append(x[j][2])
        # bgraph_lst.append(x[j][3])
    agraph = torch.cat(agraph_lst, 0)
    bgraph = torch.cat(bgraph_lst, 0)
    return [f_a, f_b, agraph, bgraph, N_atoms_scope]

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    print("index_size:",index_size)
    suffix_dim = source.size()[1:]
    print("suffix_dim:",suffix_dim)
    final_size = index_size + suffix_dim
    print("final_size:",final_size)
    print("index.view(-1):",index.view(-1))
    target = source.index_select(dim, index.view(-1))
    print("target:",target)
    return target.view(final_size)

class MPNN(nn.Sequential):

    def __init__(self, mpnn_hidden_size, mpnn_depth):
        super(MPNN, self).__init__()
        self.mpnn_hidden_size = mpnn_hidden_size
        self.mpnn_depth = mpnn_depth 

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

    def forward(self, feature):
        '''
            fatoms: (x, 39)
            fbonds: (y, 50)
            agraph: (x, 6)
            bgraph: (y, 6)
        '''
        fatoms, fbonds, agraph, bgraph, N_atoms_bond = feature
        print("fatoms:",fatoms,fatoms.shape)
        print("fbonds:",fbonds,fbonds.shape)
        print("agraph:",agraph,agraph.shape)
        print("bgraph:",bgraph,bgraph.shape)
        print("N_atoms_bond:",N_atoms_bond,N_atoms_bond.shape)
        N_atoms_scope = []
        ##### tensor feature -> matrix feature
        N_a, N_b = 0, 0 
        fatoms_lst, fbonds_lst, agraph_lst, bgraph_lst = [],[],[],[]
        for i in range(N_atoms_bond.shape[0]):
            atom_num = int(N_atoms_bond[i][0].item()) 
            bond_num = int(N_atoms_bond[i][1].item()) 

            fatoms_lst.append(fatoms[i,:atom_num,:])
            fbonds_lst.append(fbonds[i,:bond_num,:])
            agraph_lst.append(agraph[i,:atom_num,:] + N_a)
            bgraph_lst.append(bgraph[i,:bond_num,:] + N_b)

            N_atoms_scope.append((N_a, atom_num))
            N_a += atom_num 
            N_b += bond_num 
        print("fatoms_lst:",fatoms_lst)
        print("fbonds_lst:",fbonds_lst)
        print("agraph_lst:",agraph_lst)
        print("bgraph_lst:",bgraph_lst)
        print("N_atoms_scope",N_atoms_scope)
        #print(N_a, N_b)
        fatoms = torch.cat(fatoms_lst, 0)
        fbonds = torch.cat(fbonds_lst, 0)
        agraph = torch.cat(agraph_lst, 0)
        bgraph = torch.cat(bgraph_lst, 0)
        ##### tensor feature -> matrix feature
        print("fatoms:",fatoms)
        print("fbonds:",fbonds)
        print("agraph:",agraph)
        print("bgraph:",bgraph)

        agraph = agraph.long()
        bgraph = bgraph.long()    

        fatoms = create_var(fatoms).to(device)
        fbonds = create_var(fbonds).to(device)
        agraph = create_var(agraph).to(device)
        bgraph = create_var(bgraph).to(device)

        binput = self.W_i(fbonds) #### (y, d1)
        print("binput:",binput)
        message = F.relu(binput)  #### (y, d1)        
        print("message:",message)
        
        for i in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            print(i,"nei_message",nei_message)
            nei_message = nei_message.sum(dim=1)
            print(i,"nei_message_1",nei_message)
            nei_message = self.W_h(nei_message)
            print(i,"nei_message_2",nei_message)
            message = F.relu(binput + nei_message) ### (y,d1) 
            print(i,"message",message)

        nei_message = index_select_ND(message, 0, agraph)
        print("nei_message:",nei_message)
        nei_message = nei_message.sum(dim=1)
        print("nei_message_1:",nei_message)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        print("ainput:",ainput)
        atom_hiddens = F.relu(self.W_o(ainput))
        print("atom_hiddens:",atom_hiddens)
        output = [torch.mean(atom_hiddens.narrow(0, sts,leng), 0) for sts,leng in N_atoms_scope]
        print("output:",output)
        output = torch.stack(output, 0)
        print("output_1:",output)
        return output 

def out(smi):
    config = {"hidden_dim_drug": 64,
            "mpnn_depth": 3,
    }
    model = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])
    model = model.to(device)
    print("model:", model)
    ## smiles2mpnnfeature -> mpnn_collate_func -> mpnn_feature_collate_func -> MPNN.forward 
    mol_0 = smiles2mpnnfeature(smi)
    [mol_1] = mpnn_collate_func([[mol_0]])
    print("mol_1:",mol_1)
    a = model(mol_1)
    #print(a)

def main():
    smi = str(sys.argv[1])
    out(smi)

if __name__ == '__main__':
    main()
