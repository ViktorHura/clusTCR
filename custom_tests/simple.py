import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from functools import partial

from clustcr import test_func
from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, encode_data


class Refset(Dataset):
    def __init__(self, list):
        self.data = list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def encodeSequence(sequence, seqA, aa_keys, max_sequence_length=25):
    mat = np.array([aa_keys.loc[aa] for aa in sequence])
    padding = np.zeros((max_sequence_length - mat.shape[0], mat.shape[1]))
    mat = np.append(mat, padding, axis=0)
    mat = np.transpose(mat)

    mat2 = np.array([aa_keys.loc[aa] for aa in seqA])
    padding = np.zeros((max_sequence_length - mat2.shape[0], mat2.shape[1]))
    mat2 = np.append(mat2, padding, axis=0)
    mat2 = np.transpose(mat2)

    matstack = np.stack([mat, mat2])
    matstack = torch.from_numpy(matstack)
    return torch.reshape(matstack, (matstack.size(dim=1), 2, matstack.size(dim=2)))


def encode_func(sequences, keys, model, device):
    seqlist = [encodeSequence(b, a, keys) for b,a in sequences]
    data = DataLoader(Refset(seqlist), batch_size=1000, num_workers=4, shuffle=False)
    return encode_data(data, model, device)


def main():
    input_size = (31, 2, 25)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(input_size).to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # import torchinfo
    # s = torchinfo.summary(model, [input_size, input_size], batch_dim=0,
    #                   col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=0)
    # print(s)

    aa_keys = pd.read_csv('AA_keys.csv', index_col='One Letter')
    partial_func = partial(encode_func, keys=aa_keys, model=model, device=device)

    seqlist = [("ARN", "ARN") for i in range(3)]

    output = partial_func(seqlist)

    print(test_func())


if __name__ == "__main__":
    main()