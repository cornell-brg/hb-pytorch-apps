#!/usr/bin/env python

"""
    graphsage/main.py

    Note to program performers
        This is a simplfied down implementation of Workflow #46 from the August release

    The kernels are basically the same, but this shows how instances of the  GraphSAGE algorithm
    can be implemented w/ significantly less complex code.

    One difference is that each node has a feature vector here, instead of a learned embedding.
    This change was made because tester's don't necessarily have access to GPUs, and training the
    embedding layer on the CPU is too slow using standard methods, and too hard using tricks.

"""

import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from scipy.io import mmread

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(24)
torch.backends.cudnn.deterministic = True

# --
# Data loaders

def make_dataloader(idx, target, batch_size, shuffle):
    dataloader = DataLoader(
        dataset=TensorDataset(
            torch.LongTensor(idx),
            torch.FloatTensor(target)
        ),
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=4,
    )

    return dataloader

# --
# Define model

class TorchNeighborhoodSampler:
    def __init__(self, adj, batch_size, n_neibs, hammerblade=True):
        indptr  = adj.indptr.reshape(-1, 1)
        indices = adj.indices
        degrees = np.asarray(adj.sum(axis=-1)).squeeze().reshape(-1, 1)

        self.indptr  = torch.LongTensor(indptr.astype(np.int64))
        self.indices = torch.LongTensor(indices.astype(np.int64))
        self.degrees = torch.LongTensor(degrees.astype(np.int64))

#        self.max_degree = int(self.degrees.max())

        self.n_neibs = n_neibs
        self._buffer = torch.IntTensor(batch_size * n_neibs ** 2)

        self.hammerblade = hammerblade
        if self.hammerblade:
            self.indptr  = self.indptr.hammerblade()
            self.indices = self.indices.hammerblade()
            self.degrees = self.degrees.hammerblade()
            self._buffer = self._buffer.hammerblade()
 
    def sample_neighbors(self, nodes):
        self._buffer.random_()

        n_nodes = nodes.shape[0]
        neibs   = self._buffer[:(n_nodes * self.n_neibs)].view(n_nodes, self.n_neibs)
        tmp1 = self.degrees[nodes]
        tmp1 = tmp1.int()
        neibs = neibs % tmp1
        tmp2 = self.indptr[nodes]
        offsets = tmp2 + neibs
        return self.indices[offsets]


class GraphSAGEModel(nn.Module):
    def __init__(self, adj, feats, hidden_dim, output_dim, batch_size, n_neibs, hammerblade):
        super().__init__()
        print("nnz of adj:", adj.getnnz())
        self.sampler = TorchNeighborhoodSampler(
            adj=adj, batch_size=batch_size, n_neibs=n_neibs, hammerblade=hammerblade)

        self.emb = nn.Embedding(feats.shape[0], feats.shape[1])
        with torch.no_grad():
            self.emb.weight.set_(feats)
        self.emb.weight.requires_grad = False

        self.enc = nn.Sequential(
            nn.Linear(feats.shape[1], hidden_dim, bias=False),
            nn.ReLU(inplace=True),
        )

        self.hidden1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out     = nn.Linear(hidden_dim, output_dim)

        self.hidden1.bias.data.zero_()
        self.hidden2.bias.data.zero_()
        self.out.bias.data.zero_()

    def forward(self, idx0):
        n_idx   = idx0.shape[0]
        n_neibs = self.sampler.n_neibs
        idx1 = self.sampler.sample_neighbors(idx0)
        idx2 = self.sampler.sample_neighbors(idx1.view(n_idx * n_neibs)).view(n_idx, n_neibs, n_neibs)

        # encode 1-hop neighbors
        emb1 = self.emb(idx1)
        enc_1hop = self.enc(emb1)

        # encode 2-hop neighbors
        emb2 = self.emb(idx2)
        avg_2hop_intermediate = self.enc(emb2)
        avg_2hop = avg_2hop_intermediate.mean(dim=-2)

        # combine 1- and 2-hop encodings
        hidden_state1 = torch.cat([enc_1hop, avg_2hop], dim=-1)
        hidden_state1 = self.hidden1(hidden_state1)
        hidden_state1 = F.relu(hidden_state1)

        avg1_hop = hidden_state1.mean(dim=-2)

        # Another nonlinear layer
        hidden_state0 = self.hidden2(avg1_hop)
        hidden_state0 = F.relu(hidden_state0)

        prediction = self.out(hidden_state0)
        return prediction

# --
# Command line interface

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph-path', type=str, default='data/pokec.mtx')
    parser.add_argument('--meta-path',  type=str, default='data/pokec.meta')
    parser.add_argument('--feat-path',  type=str, default='data/pokec.feat.npy')

    parser.add_argument('--n-epochs',     type=int, default=3)
    parser.add_argument('--batch-size',   type=int, default=256)
    parser.add_argument('--n-neibs',      type=int, default=12)
    parser.add_argument('--hidden-dim',   type=int, default=64)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)

    parser.add_argument('--hammerblade', action="store_true")
    parser.add_argument('--seed', type=int, default=555)

    return parser.parse_args()


def set_lr(opt, lr):
    for p in opt.param_groups:
        p['lr'] = lr

# --
# CLI

args = parse_args()

_ = torch.manual_seed(args.seed + 2)
#_ = torch.hammerblade.manual_seed(args.seed + 3)

# --
# IO
# Initial Hammerblade environment
#torch.hammerblade.init()

meta  = pd.read_csv(args.meta_path, sep='\t')
adj   = mmread(args.graph_path).tocsr()
feats = torch.FloatTensor(np.load(args.feat_path))

train_meta = meta[meta.train_mask == 1]
valid_meta = meta[meta.train_mask != 1]

print("number of nodes in train set:", len(train_meta))
print("number of nodes in validation set:", len(valid_meta))

train_idx    = train_meta.node_id.values.astype(np.int64)
train_target = train_meta.target.values.astype(np.float32)

valid_idx    = valid_meta.node_id.values.astype(np.int64)
valid_target = valid_meta.target.values.astype(np.float32)

# --
# Make model

model = GraphSAGEModel(
    adj=adj,
    feats=feats,
    hidden_dim=args.hidden_dim,
    output_dim=1,
    batch_size=args.batch_size,
    n_neibs=args.n_neibs,
    hammerblade=args.hammerblade
)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True)

if args.hammerblade:
    model = model.hammerblade()

# --
# Make dataloaders

train_dataloader = make_dataloader(train_idx, train_target, args.batch_size, shuffle=True)
valid_dataloader = make_dataloader(valid_idx, valid_target, args.batch_size, shuffle=False)

# --
# Train for a few epochs
start = time()
torch.hammerblade.profiler.enable()
for epoch in range(args.n_epochs):

    # --
    # Train one epoch

    _ = model.train()

    for i, (idx, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        if args.hammerblade:
            idx, target = idx.hammerblade(), target.hammerblade()
        progress = epoch + i / len(train_dataloader)
        set_lr(opt, args.lr * (1 - progress / args.n_epochs))

        preds = model(idx).squeeze()
        loss  = F.l1_loss(preds, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
    # --
    # Predict
    _ = model.eval()

    preds = []
    for idx, _ in tqdm(valid_dataloader, total=len(valid_dataloader)):
        if args.hammerblade:
            idx = idx.hammerblade()

        pred = model(idx).squeeze()
        pred = pred.detach().cpu().numpy()
        preds.append(pred)

    preds = np.hstack(preds)

    # --
    # Compute scores

    print(json.dumps({
         "epoch" : epoch,
         "mae"   : float(np.abs(preds - valid_target).mean()),
         "corr"  : float(np.corrcoef(preds, valid_target)[0, 1]),
    }))
torch.hammerblade.profiler.disable()
print(torch.hammerblade.profiler.exec_time.fancy_print(trimming=True))
elapsed = time() - start
print("elapsed time for training 3 epochs;", elapsed, file=sys.stderr)
#print("Backward time;", backward_time, file=sys.stderr)
# --
# Save results

# os.makedirs('results', exist_ok=True)
#
# open('results/elapsed', 'w').write(str(elapsed))
# np.savetxt('results/preds', preds, fmt='%.6e')
