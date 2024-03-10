import time
import pandas as pd

def create_edgelist(cdr3, filename=None):
    '''
    Create tab-separated edgelist of edges with HD = 1, from a set of sequences.    
    '''
    # Set makes sure there are no dupes
    cdr3 = set(cdr3)
    
    # Hashing
    cdr3hash = dict()
    for cdr in cdr3:
        for hash in (cdr[::2], cdr[1::2]):
            if hash not in cdr3hash:
                cdr3hash[hash] = set()
            cdr3hash[hash].add(cdr)
            
    # Generate network
    edgelist = set()
    for hash in cdr3hash:
        if len(cdr3hash[hash]) >= 1:
            for cdr1 in cdr3hash[hash]:
                for cdr2 in cdr3hash[hash]:
                    if cdr1 != cdr2:
                        if cdr1 <= cdr2:
                            if sum(ch1 != ch2 for ch1, ch2 in zip(cdr1, cdr2)) == 1:
                                edgelist.add(cdr1 + "\t" + cdr2)
    
    # Save edgelist to file
    if filename is not None:
        with open(filename, 'w') as f:
            for edge in edgelist:
                f.write('%s\n' % edge)

    return edgelist

from scipy.spatial import KDTree

def create_edgelist_mod(cdr3, extended, model):
    '''
    Create tab-separated edgelist of edges with HD = 1, from a set of sequences.
    '''
    # Set makes sure there are no dupes
    cdr3 = set(cdr3)

    data = pd.DataFrame(data=cdr3)
    dat = pd.merge(data, extended, on=0, how='inner').drop_duplicates()

    encodings = model(dat[['CDR3_beta', 'CDR3_alpha']])

    tree = KDTree(data=encodings)

    pairs = tree.query_pairs(0.18, p=2)

    edgelist = set()
    for idx1, idx2 in pairs:
        cdr1 = dat[0].iloc[idx1]
        cdr2 = dat[0].iloc[idx2]
        edgelist.add(cdr1 + "\t" + cdr2)

    return edgelist


def create_edgelist_vgene(clusters, filename=None):
    '''
    Create tab-separated edgelist of edges with HD = 1, from a set of sequences.    
    '''
    
    # Set makes sure there are no dupes
    tcrs = [(clusters.iloc[i]["junction_aa"], 
             clusters.iloc[i]["v_call"]) for i in range(len(clusters))]
    tcrs = set(tcrs)
    
    # Hashing
    cdr3hash = dict()
    for tcr in tcrs:
        cdr = tcr[0]
        for hash in (cdr[::2], cdr[1::2]):
            if hash not in cdr3hash:
                cdr3hash[hash] = set()
            cdr3hash[hash].add(tcr)
            
    # Generate network
    edgelist = set()
    for hash in cdr3hash:
        if len(cdr3hash[hash]) >= 1:
            for tcr1 in cdr3hash[hash]:
                for tcr2 in cdr3hash[hash]:
                    if tcr1 != tcr2:
                        if tcr1[0] <= tcr2[0]:
                            if sum(ch1 != ch2 for ch1, ch2 in zip(tcr1[0], tcr2[0])) <= 1:
                                edgelist.add((tcr1,tcr2))
    
    edges = pd.DataFrame(edgelist, columns=["source", "target"])
    for col in edges:
        edges[col] = edges[col].apply(lambda x: "_".join(x))
    
    # # Save edgelist to file
    # if filename is not None:
    #     with open(filename, 'w') as f:
    #         for edge in edgelist:
    #             f.write('%s\n' % edge)

    return edges
    
def timeit(myfunc):
    # Decorator to keep track of time required to run a function
    def timed(*args, **kwargs):
        start = time.time()
        result = myfunc(*args, **kwargs)
        end = time.time()
        print(f'Total time to run ClusTCR: {(end-start):.3f}s')
        return result
    return timed
