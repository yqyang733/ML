import sys, os
from collections import defaultdict
import pandas as pd
import numpy as np
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import SequentialSampler
import copy
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
import matplotlib.pyplot as plt
import pickle

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
MAX_ATOM = 400
MAX_BOND = MAX_ATOM * 2
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
MAX_SEQ_PROTEIN = 1000
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

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

def sampling(DTI, dict_icv_drug_encoding, dict_target_protein_encoding):
    target_set = list(dict_target_protein_encoding.keys())
    with open(DTI) as f:
        f1 = f.readlines()
    dti_mutidict = defaultdict(list)
    [dti_mutidict[i.split(",")[0]].append(i.split(",")[1].strip()) for i in f1]
    whole_positive = []
    whole_negetive = []
    for key in dti_mutidict:
        for i in dti_mutidict[key]:
            whole_positive.append([key,i,dict_icv_drug_encoding[key],dict_target_protein_encoding[i],1])
        target_no = target_set[:]
        [target_no.remove(i) for i in dti_mutidict[key]]
        for a in target_no:
            whole_negetive.append([key,a,dict_icv_drug_encoding[key],dict_target_protein_encoding[a],0])
    #print("whole_positive: ",whole_positive[0])
    #print("whole_negetive: ",whole_negetive[0])
    whole_positive = np.array(whole_positive,dtype=object)
    whole_negetive = np.array(whole_negetive,dtype=object)
    whole_positive_muti_10 = np.tile(whole_positive,(10,1))
    #print(len(whole_positive),len(whole_positive_muti_10))
    #[print(i) for i in whole_positive_muti_10[:,0]]
    np.random.seed(10)
    whole_negetive_index = np.random.choice(np.arange(len(whole_negetive)), size=10 * len(whole_positive),replace=False)
    whole_negetive_1 = np.array([whole_negetive[i] for i in whole_negetive_index])
    # print(len(whole_positive_muti_10),len(whole_negetive_1))
    # print(whole_negetive_1[:,0])
    data_set = np.vstack((whole_positive_muti_10,whole_negetive_1))
    #print(len(data_set),data_set[0])

    #all_fold_dataset = []
    fold_dataset = []
    #print(data_set[:,0:4][0:2],data_set[:,4][0:2])
    #print(type(data_set[:, 4]))
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    print((len(data_set)))
    #temp = np.array([[i,1] for i in range(len(data_set))])
    #print(len(temp))
    #label_encoder = LabelEncoder()
    y = LabelEncoder().fit_transform(data_set[:, 4])
    for train, test in kf.split(data_set[:, 0:4], y):
        train_data, test_data = data_set[train], data_set[test]
        one_fold_dataset = []
        one_fold_dataset.append(train_data)
        one_fold_dataset.append(test_data)
        fold_dataset.append(one_fold_dataset)
    # print(len(fold_dataset[0][0]))
    # print(len(fold_dataset[0][1]))
    # print((fold_dataset[0][0][0]))
    # output = pd.DataFrame(fold_dataset[0][0])
    # print(output)
    return fold_dataset

def get_train_test(fold_1):
    train = pd.DataFrame(fold_1[0])
    train.rename(columns={0:'ICV',
                        1: 'Target',
                        2: 'drug_encoding',
                        3: 'target_encoding',
                        4: 'label'}, 
                        inplace=True)
    test = pd.DataFrame(fold_1[1])
    test.rename(columns={0:'ICV',
                        1: 'Target',
                        2: 'drug_encoding',
                        3: 'target_encoding',
                        4: 'label'}, 
                        inplace=True)
    return train, test

def generate_config(result_folder = "./result/",
                    input_dim_drug = 1024,
                    input_dim_protein = 8420,
                    hidden_dim_drug = 256, 
                    hidden_dim_protein = 256,
                    cls_hidden_dims = [1024, 1024, 512],
                    batch_size = 256,
                    train_epoch = 10,
                    test_every_X_epoch = 20,
                    LR = 1e-4,
                    decay = 0,
                    mpnn_hidden_size = 50,
                    mpnn_depth = 3,
                    cnn_target_filters = [32,64,96],
                    cnn_target_kernels = [4,8,12],
                    num_workers = 0):
    base_config = {'input_dim_drug': input_dim_drug,
                    'input_dim_protein': input_dim_protein,
                    'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
                    'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
                    'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
                    'batch_size': batch_size,
                    'train_epoch': train_epoch,
                    'test_every_X_epoch': test_every_X_epoch, 
                    'LR': LR,
                    'result_folder': result_folder,
                    'binary': False,
                    'num_workers': num_workers
    }
    if not os.path.exists(base_config['result_folder']):
        os.makedirs(base_config['result_folder'])
    base_config['hidden_dim_drug'] = hidden_dim_drug
    base_config['batch_size'] = batch_size 
    base_config['mpnn_hidden_size'] = mpnn_hidden_size
    base_config['mpnn_depth'] = mpnn_depth
    base_config['cnn_target_filters'] = cnn_target_filters
    base_config['cnn_target_kernels'] = cnn_target_kernels
    return base_config

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

class MPNN(nn.Sequential):

    def __init__(self, mpnn_hidden_size, mpnn_depth):
        super(MPNN, self).__init__()
        self.mpnn_hidden_size = mpnn_hidden_size
        self.mpnn_depth = mpnn_depth 
        from DeepPurpose.chemutils import ATOM_FDIM, BOND_FDIM

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

    def forward(self, feature):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''
            fatoms: (x, 39)
            fbonds: (y, 50)
            agraph: (x, 6)
            bgraph: (y, 6)
        '''
        fatoms, fbonds, agraph, bgraph, N_atoms_bond = feature
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


        fatoms = torch.cat(fatoms_lst, 0)
        fbonds = torch.cat(fbonds_lst, 0)
        agraph = torch.cat(agraph_lst, 0)
        bgraph = torch.cat(bgraph_lst, 0)
        ##### tensor feature -> matrix feature


        agraph = agraph.long()
        bgraph = bgraph.long()  

        fatoms = create_var(fatoms).to(device)
        fbonds = create_var(fbonds).to(device)
        agraph = create_var(agraph).to(device)
        bgraph = create_var(bgraph).to(device)

        binput = self.W_i(fbonds) #### (y, d1)
        message = F.relu(binput)  #### (y, d1)      

        for i in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message) ### (y,d1) 

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))
        output = [torch.mean(atom_hiddens.narrow(0, sts,leng), 0) for sts,leng in N_atoms_scope]
        output = torch.stack(output, 0)
        return output 

class CNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if encoding == 'drug':
            in_ch = [63] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))
            #n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

        if encoding == 'protein':
            in_ch = [26] + config['cnn_target_filters']
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double().to(device)
            n_size_p = self._get_conv_output((26, 1000))

            self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

    def _get_conv_output(self, shape):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double().to(device))
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #x = v.float().to(device)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #x = x.float().to(device)
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v = self._forward_features(v.double().to(device))
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_protein, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.input_dim_protein = config['hidden_dim_protein']

        self.model_drug = model_drug
        self.model_protein = model_protein

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]
        
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_protein(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i==(len(self.predictor)-1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f

def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

class data_process_loader(data.Dataset):

    def __init__(self, list_IDs, labels, df, **config):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df
        self.config = config

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['drug_encoding']        
        v_p = self.df.iloc[index]['target_encoding']
        v_p = protein_2_embed(v_p)
        y = self.labels[index]
        return v_d, v_p, y

def mpnn_feature_collate_func(x):
    N_atoms_scope = torch.cat([i[4] for i in x], 0)
    f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
    f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
    agraph_lst, bgraph_lst = [], []
    for j in range(len(x)):
        agraph_lst.append(x[j][2].unsqueeze(0))
        bgraph_lst.append(x[j][3].unsqueeze(0))
    agraph = torch.cat(agraph_lst, 0)
    bgraph = torch.cat(bgraph_lst, 0)
    return [f_a, f_b, agraph, bgraph, N_atoms_scope]

def mpnn_collate_func(x):
    mpnn_feature = [i[0] for i in x]
    mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
    from torch.utils.data.dataloader import default_collate
    x_remain = [list(i[1:]) for i in x]
    x_remain_collated = default_collate(x_remain)
    return [mpnn_feature] + x_remain_collated

def save_dict(path, obj):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class DBTA:
    def __init__(self, **config):

        self.model_drug = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])
        self.model_protein = CNN('protein', **config)
        self.model = Classifier(self.model_drug, self.model_protein, **config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = device
        self.result_folder = config['result_folder']
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)            
        self.binary = False
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0

    def test_(self, data_generator, model, repurposing_mode = False, test = False):
        y_pred = []
        y_label = []
        model.eval()
        for i, (v_d, v_p, label) in enumerate(data_generator):
            v_d = v_d
            score = self.model(v_d, v_p)
            if self.binary:
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score)).detach().cpu().numpy()
            else:
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
                #loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(device))
                logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        model.train()
        if self.binary:
            if repurposing_mode:
                return y_pred
            ## ROC-AUC curve
            if test:
                roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
                plt.figure(0)
                roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding + '_' + self.target_encoding)
                plt.figure(1)
                pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
                prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding + '_' + self.target_encoding)

            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred
        else:
            if repurposing_mode:
                return y_pred
            return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred, loss

    def train(self, train, val, test = None, verbose = True):
        if len(train.label.unique()) == 2:
            #print("binary")
            self.binary = True
            self.config['binary'] = True

        lr = self.config['LR']
        decay = self.config['decay']
        BATCH_SIZE = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        if 'test_every_X_epoch' in self.config.keys():
            test_every_X_epoch = self.config['test_every_X_epoch']
        else:     
            test_every_X_epoch = 40
        loss_history = []

        self.model = self.model.to(self.device)
        #self.model = self.model.to(device)

        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = nn.DataParallel(self.model, dim = 0)
        elif torch.cuda.device_count() == 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            if verbose:
                print("Let's use CPU/s!")
        # Future TODO: support multiple optimizers with parameters
        opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
        if verbose:
            print('--- Data Preparation ---')

        params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': self.config['num_workers'],
                'drop_last': False,
                'collate_fn': mpnn_collate_func
                }

        training_generator = data.DataLoader(data_process_loader(train.index.values, train.label.values, train, **self.config), **params)
        validation_generator = data.DataLoader(data_process_loader(val.index.values, val.label.values, val, **self.config), **params)
        
        if test is not None:
            info = data_process_loader(test.index.values, test.Label.values, test, **self.config)
            params_test = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': self.config['num_workers'],
                    'drop_last': False,
                    'sampler':SequentialSampler(info),
                    'collate_fn': mpnn_collate_func
                    }
            testing_generator = data.DataLoader(data_process_loader(test.index.values, test.label.values, test, **self.config), **params_test)

        # early stopping
        if self.binary:
            max_auc = 0
        else:
            max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"] 
        if self.binary:
            valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
        else:
            valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
        table = PrettyTable(valid_metric_header)
        float2str = lambda x:'%0.4f'%x
        if verbose:
            print('--- Go for Training ---')
        writer = SummaryWriter()
        t_start = time() 
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                v_p = v_p.float().to(self.device) 
                #v_p = v_p.float().to(device) 
                v_d = v_d
               
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
                #label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

                if self.binary:
                    loss_fct = torch.nn.BCELoss()      # 计算目标值和预测值之间的二进制交叉熵损失函数
                    m = torch.nn.Sigmoid()    # 二分类问题，逻辑回归。
                    n = torch.squeeze(m(score), 1)
                    loss = loss_fct(n, label)
                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(score, 1)
                    loss = loss_fct(n, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 100 == 0):
                        t_now = time()
                        print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
                            ". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
                        ### record total run time
                        

            ##### validate, select the best model up to now 
            with torch.set_grad_enabled(False):
                if self.binary:  
                    ## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
                    auc, auprc, f1, loss, logits = self.test_(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
                    valid_metric_record.append(lst)
                    if auc > max_auc:
                        model_max = copy.deepcopy(self.model)
                        max_auc = auc   
                    if verbose:
                        print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
                          ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                          str(loss)[:7])
                else:  
                    ### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
                    mse, r2, p_val, CI, logits, loss_val = self.test_(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
                    valid_metric_record.append(lst)
                    if mse < max_MSE:
                        model_max = copy.deepcopy(self.model)
                        max_MSE = mse
                    if verbose:
                        print('Validation at Epoch '+ str(epo + 1) + ' with loss:' + str(loss_val.item())[:7] +', MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
                         + str(r2)[:7] + ' with p-value: ' + str(p_val)[:7] +' , Concordance Index: '+str(CI)[:7])
                        writer.add_scalar("valid/mse", mse, epo)
                        writer.add_scalar("valid/pearson_correlation", r2, epo)
                        writer.add_scalar("valid/concordance_index", CI, epo)
                        writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
            table.add_row(lst)

        # load early stopped model
        self.model = model_max

        #### after training 
        prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
        with open(prettytable_file, 'w') as fp:
            fp.write(table.get_string())

        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
            if self.binary:
                auc, auprc, f1, loss, logits = self.test_(testing_generator, model_max, test = True)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [auc, auprc, f1])))
                if verbose:
                    print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                      ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                      str(loss)[:7])                
            else:
                mse, r2, p_val, CI, logits, loss_test = self.test_(testing_generator, model_max)
                test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
                test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
                if verbose:
                    print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                      + ' with p-value: ' + str(p_val) +' , Concordance Index: '+str(CI))
            np.save(os.path.join(self.result_folder, str(self.drug_encoding) + '_' + str(self.target_encoding) 
                     + '_logits.npy'), np.array(logits))                
    
            ######### learning record ###########

            ### 1. test results
            prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        ### 2. learning curve 
        fontsize = 16
        iter_num = list(range(1,len(loss_history)+1))
        plt.figure(3)
        plt.plot(iter_num, loss_history, "bo-")
        plt.xlabel("iteration", fontsize = fontsize)
        plt.ylabel("loss value", fontsize = fontsize)
        pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        fig_file = os.path.join(self.result_folder, "loss_curve.png")
        plt.savefig(fig_file)
        if verbose:
            print('--- Training Finished ---')
            writer.flush()
            writer.close()
          

    def predict(self, df_data):
        '''
            utils.data_process_repurpose_virtual_screening 
            pd.DataFrame
        '''
        print('predicting...')
        info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
        self.model.to(device)
        params = {'batch_size': self.config['batch_size'],
                'shuffle': False,
                'num_workers': self.config['num_workers'],
                'drop_last': False,
                'sampler':SequentialSampler(info),
                'collate_fn': mpnn_collate_func}

        generator = data.DataLoader(info, **params)

        score = self.test_(generator, self.model, repurposing_mode = True)
        return score

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
        #if device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location = torch.device('cpu'))
        # to support training from multi-gpus data-parallel:
        
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config['binary']

def model_initialize(**config):
    model = DBTA(**config)
    return model

def fold_10_valid(fold_datase):
    for i in range(len(fold_datase)):
        print("the "+str(i)+" cross fold beginning!")
        train, tes = get_train_test(fold_datase[i])
        config = generate_config(cls_hidden_dims = [1024,1024,512],
                         result_folder = "./result_" + str(i) + "/",
                         train_epoch = 30, 
                         LR = 0.0001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )
        model = model_initialize(**config)
        #model = model.cuda()
        model.train(train, tes)
        model.save_model("./result_" + str(i) + "/fold_" + str(i) + "_model")

def main():
    smiles = str(sys.argv[1])
    protein = str(sys.argv[2])
    #protein = str(sys.argv[1])
    DTI = str(sys.argv[3])
    dict_icv_drug_encoding = smiles_embed(smiles)
    dict_target_protein_encoding = protein_embed(protein)
    fold_dataset = sampling(DTI, dict_icv_drug_encoding, dict_target_protein_encoding)
    fold_10_valid(fold_dataset)

if __name__=="__main__":
    main() 