import math

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from functools import partial

from clustcr import test_func
from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, encode_data
from clustcr import datasets
import swifter
from scipy.spatial import distance
import matplotlib.pyplot as plt


class Refset(Dataset):
    def __init__(self, list):
        self.data = list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def encodeSequence(sequence, seqA, aa_keys, max_sequence_length=25):
    mat = np.array([aa_keys.loc[aa] for aa in sequence[:max_sequence_length]])
    padding = np.zeros((max_sequence_length - mat.shape[0], mat.shape[1]))
    mat = np.append(mat, padding, axis=0)
    mat = np.transpose(mat)

    mat2 = np.array([aa_keys.loc[aa] for aa in seqA[:max_sequence_length]])
    padding = np.zeros((max_sequence_length - mat2.shape[0], mat2.shape[1]))
    mat2 = np.append(mat2, padding, axis=0)
    mat2 = np.transpose(mat2)

    matstack = np.stack([mat, mat2])
    matstack = torch.from_numpy(matstack)
    return torch.reshape(matstack, (matstack.size(dim=1), 2, matstack.size(dim=2)))


def encode_func(sequences, keys, model, device):
    l = sequences.swifter.apply(lambda x: encodeSequence(x[0], x[1], keys), axis=1).to_list()
    data = DataLoader(Refset(l), batch_size=1000, num_workers=4, shuffle=False)
    return encode_data(data, model, device)


def plot_distances(sim, dissim, n_bins, scale, title="Distance distributions"):
    fig, axs = plt.subplots(3, 1, sharex='col', tight_layout=True, figsize=([20, 9.6]))
    plt.xlim(scale)
    axs[0].hist(sim, bins=n_bins, weights=np.ones(len(sim)) / len(sim))
    axs[1].hist(dissim, bins=n_bins, weights=np.ones(len(dissim)) / len(dissim))
    axs[2].boxplot([dissim, sim], vert=False)
    axs[2].set_yticklabels(["negative", "positive"])

    fig.suptitle(title, fontsize="xx-large", fontweight="bold")
    axs[0].set_ylabel("Positive pair")
    axs[1].set_ylabel("Negative pair")
    plt.xlabel("Distance")


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

    vdjdb = datasets.vdjdb_paired(epitopes=True).sample(n=1000, random_state=42)
    out = partial_func(vdjdb[['CDR3_beta', 'CDR3_alpha']])
    vdjdb.insert(0, 'Encoding', out)
    vdjdb.drop(columns=['CDR3_alpha', 'CDR3_beta'], inplace=True)

    pairs = vdjdb.merge(vdjdb, how='cross')
    pairs = pairs.query("(Encoding_x > Encoding_y)")

    pairs['Distance'] = pairs.swifter.apply(lambda x: distance.euclidean(x['Encoding_x'], x['Encoding_y']), axis=1)
    pairs.drop(columns=['Encoding_x', 'Encoding_y'], inplace=True)

    sim = pairs[pairs['Epitope_x'] == pairs['Epitope_y']]['Distance'].to_list()
    dsim = pairs[pairs['Epitope_x'] != pairs['Epitope_y']]['Distance'].to_list()

    n_bins = math.ceil(math.sqrt(len(sim)+len(dsim)))
    scale = (0, pairs['Distance'].max())
    plot_distances(sim, dsim, n_bins, scale)
    plt.show()

    #1.23


if __name__ == "__main__":
    main()