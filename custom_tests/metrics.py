import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from functools import partial

from clustcr import Clustering, datasets
from clustcr import test_func

plt.style.use(['seaborn-v0_8-white', 'seaborn-v0_8-paper'])
plt.rc('font', family='serif')
sns.set_palette('Set1')
sns.set_context('paper', font_scale=1.3)

MIN_SAMPLE = 1000
MAX_SAMPLE = 20000
STEP_SIZE = 1000

sample_sizes = [x for x in range(MIN_SAMPLE,
                     MAX_SAMPLE,
                     STEP_SIZE)] + [MAX_SAMPLE]


import torch
from simple import encode_func, SiameseNetwork


def main():
    # Import data set with known antigen specificities
    vdjdb = datasets.vdjdb_paired(epitopes=True)

    input_size = (31, 2, 25)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(input_size).to(device)
    model.load_state_dict(torch.load('model_ct2.pt'))
    model.eval()

    aa_keys = pd.read_csv('AA_keys.csv', index_col='One Letter')
    partial_func = partial(encode_func, keys=aa_keys, model=model, device=device)

    # Initiate output dataframe
    output = pd.DataFrame()

    # Perform analysis for different sample sizes
    for s in sample_sizes:
        # Create sample with size s
        sample = vdjdb.sample(s, random_state=42)
        cdr3, alpha = sample['CDR3_beta'], sample['CDR3_alpha']

        # Perform clustering
        t3 = time.time()
        ts3 = Clustering().fit(cdr3, alpha=sample['CDR3_alpha'], metric='levenshtein')
        ts3_out = ts3.metrics(sample).summary()
        t3 = time.time() - t3
        ts3_out['method'] = 'Blosum62'
        ts3_out['n'] = s
        ts3_out['t'] = t3
        output = output._append(ts3_out)

        t0 = time.time()
        ts = Clustering().fit(cdr3, alpha=sample['CDR3_alpha'])
        ts_out = ts.metrics(sample).summary()
        t = time.time() - t0
        ts_out['method'] = 'Default'
        ts_out['n'] = s
        ts_out['t'] = t
        output = output._append(ts_out)

        t1 = time.time()
        ts2 = Clustering(method='two-step-mod').fit(cdr3, alpha=sample['CDR3_alpha'], model=partial_func)
        # Two-step (ClusTCR)
        ts2_out = ts2.metrics(sample).summary()
        t2 = time.time() - t1
        ts2_out['method'] = 'Siamese'
        ts2_out['n'] = s
        ts2_out['t'] = t2
        output = output._append(ts2_out)

    # Write output to file
    output.to_csv('clustcr_step_evaluation.tsv', sep='\t', index=False)

    data = output

    colors = sns.color_palette('Set1')

    retent = data[data['metrics'] == 'retention']
    purity = data[data['metrics'] == 'purity']
    pur_90 = data[data['metrics'] == 'purity_90']
    consist = data[data['metrics'] == 'consistency']

    fig = plt.figure()

    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    x = data.n.unique()

    ax1.plot(x, retent[retent['method'] == 'Default'].actual, color=colors[2], label='Default')
    ax1.plot(x, retent[retent['method'] == 'Siamese'].actual, color=colors[3], label='Siamese')
    ax1.plot(x, retent[retent['method'] == 'Blosum62'].actual, color=colors[4], label='Blosum62')
    ax1.set_ylabel('Retention')


    ax2.plot(x, purity[purity['method'] == 'Default'].actual, color=colors[2], label='Default')
    ax2.plot(x, purity[purity['method'] == 'Siamese'].actual, color=colors[3], label='Siamese')
    ax2.plot(x, purity[purity['method'] == 'Blosum62'].actual, color=colors[4], label='Blosum62')

    ax2.plot(x, purity[purity['method'] == 'Default'].baseline, color=colors[2], ls='--', lw=.5,
             label='Default - permuted')
    ax2.plot(x, purity[purity['method'] == 'Siamese'].baseline, color=colors[3], ls='--', lw=.5,
             label='Siamese - permuted')
    ax2.plot(x, purity[purity['method'] == 'Blosum62'].baseline, color=colors[4], ls='--', lw=.5,
             label='Blosum62 - permuted')
    ax2.set_ylabel('Purity')


    ax3.plot(x, pur_90[pur_90['method'] == 'Default'].actual, color=colors[2], label='Default')
    ax3.plot(x, pur_90[pur_90['method'] == 'Siamese'].actual, color=colors[3], label='Siamese')
    ax3.plot(x, pur_90[pur_90['method'] == 'Blosum62'].actual, color=colors[4], label='Blosum62')
    ax3.plot(x, pur_90[pur_90['method'] == 'Default'].baseline, color=colors[2], ls='--', lw=.5,
             label='Default - permuted')
    ax3.plot(x, pur_90[pur_90['method'] == 'Siamese'].baseline, color=colors[3], ls='--', lw=.5,
             label='Siamese - permuted')
    ax3.plot(x, pur_90[pur_90['method'] == 'Blosum62'].baseline, color=colors[4], ls='--', lw=.5,
             label='Blosum62 - permuted')
    ax3.set_ylabel(r'$f_{purity > 0.90}$')


    ax4.plot(x, consist[consist['method'] == 'Default'].actual, color=colors[2], label='Default')
    ax4.plot(x, consist[consist['method'] == 'Siamese'].actual, color=colors[3], label='Siamese')
    ax4.plot(x, consist[consist['method'] == 'Blosum62'].actual, color=colors[4], label='Blosum62')
    ax4.plot(x, consist[consist['method'] == 'Default'].baseline, color=colors[2], ls='--', lw=.5,
             label='Default - permuted')
    ax4.plot(x, consist[consist['method'] == 'Siamese'].baseline, color=colors[3], ls='--', lw=.5,
             label='Siamese - permuted')
    ax4.plot(x, consist[consist['method'] == 'Blosum62'].baseline, color=colors[4], ls='--', lw=.5,
             label='Blosum62 - permuted')
    ax4.set_ylabel('Consistency')


    ax5.plot(x, consist[consist['method'] == 'Default'].t, color=colors[2], label='Default')
    ax5.plot(x, consist[consist['method'] == 'Siamese'].t, color=colors[3], label='Siamese')
    ax5.plot(x, consist[consist['method'] == 'Blosum62'].t, color=colors[4], label='Blosum62')
    ax5.set_ylabel('t (seconds)')
    ax5.set_xlabel('n sequences')

    fig.subplots_adjust(top=1.1, hspace=1, wspace=.5)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',
               bbox_to_anchor=(1.40, 0.6))

    ax1.text(-0.25, 1.50, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax2.text(-0.25, 1.50, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax3.text(-0.25, 1.50, 'C', transform=ax3.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax4.text(-0.25, 1.50, 'D', transform=ax4.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax5.text(-0.1, 1.50, 'E', transform=ax5.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    fig.savefig('clustcr_step_evaluation-0.00.png', format='png', bbox_inches='tight')

    print(test_func())


if __name__ == "__main__":
    main()