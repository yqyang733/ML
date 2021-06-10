import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F

amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
MAX_SEQ_PROTEIN = 1000

enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
    else:
        temp = temp [:MAX_SEQ_PROTEIN]
    return temp

def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

class CNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
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
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))

            self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

def out(fas):
    config = {"cnn_target_filters": [32,64,96],
            "cnn_target_kernels": [4,8,12],
            "hidden_dim_protein": 256
    }
    model = CNN("protein", **config)
    model = model.to(device)
    print("model:", model)
    fas = trans_protein(fas)
    v_p = protein_2_embed(fas)
    v_p = torch.Tensor([v_p]).float().to(device)
    print("v_p:",v_p)
    tmp = model(v_p)
    print(tmp)

def main():
    fas = str(sys.argv[1])
    out(fas)

if __name__ == '__main__':
    main()
