#!/usr/bin/env python

"""
    lgc/main.py
    
    Note to program performers:
        - parallel_pr_nibble produces the same results as ligra's `apps/localAlg/ACL-Sync-Local-Opt.C`
        - ista produces the same results as LocalGraphClustering's `ista_dinput_dense` method
"""

import os
import sys
import argparse
import numpy as np
import torch
from time import time
from tqdm import tqdm
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr
import json

# ISTA algorithm

def ista(seeds, adj, alpha, rho, iters):
    out = []
    # Compute degree vectors/matrices
    d       = np.asarray(adj.sum(axis=-1)).squeeze()
    d_sqrt  = np.sqrt(d)
    dn_sqrt = 1 / d_sqrt

    D       = sparse.diags(d)
    Dn_sqrt = sparse.diags(dn_sqrt)
    # Normalized adjacency matrix
    Q = D - ((1 - alpha) / 2) * (D + adj)
    Q = Dn_sqrt @ Q @ Dn_sqrt

    for seed in tqdm(seeds):
        # Make personalized distribution
        s = np.zeros(adj.shape[0])
        s[seed] = 1
        # Initialize
        q = np.zeros(adj.shape[0], dtype=np.float64)
        rad   = rho * alpha * d_sqrt
        grad0 = -alpha * dn_sqrt * s
        grad  = grad0
        # Run
        for _ in range(iters):
            q    = q - grad - rad
            q    = np.maximum(q, 0)
            tmp = Q @ q
            grad = grad0 + tmp

        out.append(q * d_sqrt)
    
    return np.column_stack(out)

def spy_sparse2torch_sparse(data):
    row = data.shape[0]
    col = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = np.append([coo_data.row], [coo_data.col], axis=0)
    indices = torch.from_numpy(indices).long()
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), torch.Size([row, col]))
    return t

def test_torch_ista(seeds, adj, size, alpha, rho, iters):
    out = torch.empty(0)

    # Compute degree vectors/matrices
    d       = np.asarray(adj.sum(axis=-1)).squeeze()
    d_sqrt  = np.sqrt(d)
    dn_sqrt = 1 / d_sqrt

    D       = sparse.diags(d)
    Dn_sqrt = sparse.diags(dn_sqrt)

    # Normalized adjacency matrix
    Q = D - ((1 - alpha) / 2) * (D + adj)
    Q = Dn_sqrt @ Q @ Dn_sqrt

    # Convert numpy float64 data to torch float32 tensor
    Q = spy_sparse2torch_sparse(Q)
    d_sqrt = torch.from_numpy(d_sqrt).float()
    dn_sqrt = torch.from_numpy(dn_sqrt).float()
    zero = torch.zeros(size)
    if args.hammerblade:
        Q = Q.hammerblade()
        d_sqrt = d_sqrt.hammerblade()
        dn_sqrt = dn_sqrt.hammerblade()
        zero = zero.hammerblade()

    for seed in tqdm(seeds):
        s = np.zeros(adj.shape[0])
        s[seed] = 1
        s = torch.from_numpy(s).float()
        if args.hammerblade:
            s = s.hammerblade()
        q = zero       
        rad = rho * alpha * d_sqrt
        grad0 = -alpha * dn_sqrt * s
        grad = grad0.clone()
        for _ in range(iters):
            q = torch.max(q - grad - rad, zero)
            temp = torch.mv(Q, q)
            grad = grad + temp
        temp = torch.mul(q, d_sqrt).cpu()
        temp = temp.view(size, 1)
        out = torch.cat((out, temp), 1)
        
    return out

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pnib-epsilon', type=float, default=1e-6)
    parser.add_argument('--ista-rho', type=float, default=1e-5)
    parser.add_argument('--ista-iters', type=int, default=50)
    parser.add_argument('--hammerblade', action="store_true")
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.pnib_epsilon == 1e-6
    assert args.ista_rho == 1e-5
    assert args.ista_iters == 50
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
#    torch.hammerblade.init()
    array = mmread('data/jhu.mtx')
    adj = array.tocsr()
    torchadj = torch.from_numpy(array.toarray()).to_sparse()
    size = torchadj.size(0)
    
    # ISTA: Faster algorithm, so use more seeds to get roughly comparable total runtime
    ista_seeds = list(range(10 * args.num_seeds))
    
    # --
    # Run ISTA
    
    ista_scores = ista(ista_seeds, adj, alpha=args.alpha, rho=args.ista_rho, iters=args.ista_iters)
    assert ista_scores.shape[0] == adj.shape[0]
    assert ista_scores.shape[1] == len(ista_seeds)

#    with open('lgc_ista.json') as route:
#        data = json.load(route)
#    torch.hammerblade.profiler.route.set_route_from_json(data)
    torch.hammerblade.profiler.enable()
    torch_out = test_torch_ista(ista_seeds, adj, size, alpha=args.alpha, rho=args.ista_rho, iters=args.ista_iters)
    torch.hammerblade.profiler.disable()
    print(torch.hammerblade.profiler.exec_time.fancy_print(trimming=True))
    np_out = torch.from_numpy(ista_scores).float()
    assert torch.allclose(torch_out, np_out)
    
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    #np.savetxt('results/pnib_score.txt', pnib_scores)
    np.savetxt('results/ista_score.txt', ista_scores)
    
    #open('results/pnib_elapsed', 'w').write(str(pnib_elapsed))
    open('results/ista_elapsed', 'w').write(str(ista_elapsed))

