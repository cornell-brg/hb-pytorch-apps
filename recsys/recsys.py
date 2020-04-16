#!/bin/python
#=========================================================================
# pytorch-recsys
# adapted from darpa-sdh-prog-eval/recsys
#=========================================================================

import sys
from os import path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import argparse
import random
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nepoch', default=-1, type=int,
                    help="number of training epochs")
parser.add_argument('--nbatch', default=-1, type=int,
                    help="number of training/inference batches")
parser.add_argument('--hammerblade', default=False, action='store_true',
                    help="run MLP MNIST on HammerBlade")
parser.add_argument('--training', default=False, action='store_true',
                    help="run training phase")
parser.add_argument('--inference', default=False, action='store_true',
                    help="run inference phase")
parser.add_argument("-v", "--verbose", default=0, action='count',
                    help="increase output verbosity")
parser.add_argument("--save-model", default=False, action='store_true',
                    help="save trained model to file")
parser.add_argument("--load-model", default=False, action='store_true',
                    help="load trained model from file")
parser.add_argument('--model-filename', default="trained_model", type=str,
                    help="filename of the saved model")
parser.add_argument('--seed', default=42, type=int,
                    help="manual random seed")
parser.add_argument("--dry", default=False, action='store_true',
                    help="dry run")

# ------- workload specific options -----------
parser.add_argument('--dataset-path', type=str, default='data/ml-10m.ratings.dat',
                    help="path to raw movielens dataset")
parser.add_argument('--cache-path', type=str, default='data/cache',
                    help="path to preprocessed data cache")

args = parser.parse_args()

# By default, we do both training and inference
if (not args.training) and (not args.inference):
  args.training = True
  args.inference = True

# If not specified, run 30 epochs
if args.nepoch == -1:
  args.nepoch = 30

# If nbatch is set, nepoch is forced to be 1
if args.nbatch == -1:
  args.nbatch = 65535
else:
  args.nepoch = 1

torch.manual_seed(args.seed)
np.random.seed(args.seed + 1)
random.seed(args.seed + 2)

#-------------------------------------------------------------------------
# Prepare Dataset
#-------------------------------------------------------------------------

# Check if cache already exists
if path.exists(args.cache_path + '_train.npy') and path.exists(args.cache_path + '_valid.npy'):
  print('Cache exists ... skip dataset preprocessing', file=sys.stderr)
else:
  print('loading %s' % args.dataset_path, file=sys.stderr)
  # IO
  edges = pd.read_csv(args.dataset_path, header=None, sep='::', engine='python')[[0, 1]]
  edges.columns = ('userId', 'movieId')

  # Remap IDs to sequential integers
  uusers       = set(edges.userId)
  user_lookup  = dict(zip(uusers, range(len(uusers))))
  edges.userId = edges.userId.apply(user_lookup.get)

  umovies       = set(edges.movieId)
  movie_lookup  = dict(zip(umovies, range(len(umovies))))
  edges.movieId = edges.movieId.apply(movie_lookup.get)
  edges.movieId += 1 # Add padding character

  # Train/test split
  train, valid = train_test_split(edges, train_size=0.8, stratify=edges.userId)

  # Convert to adjacency list + save
  train_adjlist = train.groupby('userId').movieId.apply(lambda x: sorted(set(x))).values
  valid_adjlist = valid.groupby('userId').movieId.apply(lambda x: sorted(set(x))).values

  print('saving %s' % args.cache_path, file=sys.stderr)
  np.save(args.cache_path + '_train', train_adjlist)
  np.save(args.cache_path + '_valid', valid_adjlist)

# Data loaders
class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks):
        self.X      = [torch.LongTensor(xx) for xx in X]
        self.n_toks = n_toks

    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.zeros((self.n_toks,))
        y[x] += 1
        return x, y

    def __len__(self):
        return len(self.X)


def ragged_collate_fn(batch, pad_value=0):
    X, y = zip(*batch)

    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]

    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y


def make_dataloader(X, n_toks, batch_size, shuffle):
    return DataLoader(
        dataset=RaggedAutoencoderDataset(X=X, n_toks=n_toks),
        batch_size=batch_size,
        collate_fn=ragged_collate_fn,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle,
    )

# Load data
X_train = np.load('%s_train.npy' % args.cache_path, allow_pickle=True)
n_toks  = np.hstack(X_train).max() + 1

#-------------------------------------------------------------------------
# Print Layer
#-------------------------------------------------------------------------
class PrintLayer(nn.Module):

  def __init__(self):
    super(PrintLayer, self).__init__()

  def forward(self, x):
    if args.verbose > 1:
      print(x)
    return x

#-------------------------------------------------------------------------
# Autoencoder based Recsys
#-------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    def __init__(self, n_toks, emb_dim, hidden_dim, dropout, bias_offset):
        super().__init__()

        self.emb = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0

        self.act_bn_drop_1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
        )

        self.bottleneck = nn.Linear(emb_dim, hidden_dim)
        self.bottleneck.bias.data.zero_()

        self.act_bn_drop_2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(hidden_dim, n_toks)
        self.output.bias.data.zero_()
        self.output.bias.data += bias_offset

    def forward(self, x):
        x = self.emb(x).sum(dim=1)
        x = self.act_bn_drop_1(x)
        x = self.bottleneck(x)
        x = self.act_bn_drop_2(x)
        x = self.output(x)
        return x

#-------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------

model = MLPEncoder(
    n_toks=n_toks,
    emb_dim=800,
    hidden_dim=400,
    dropout=0.5,
    bias_offset=-10
)

# Load pretrained model if necessary
if args.load_model:
  model.load_state_dict(torch.load(args.model_filename))

# Move model to HammerBlade if using HB
if args.hammerblade:
  model.to(torch.device("hammerblade"))

print(model)

# Dump configs
if args.verbose > 0:
  print(args)

# Quit here if dry run
if args.dry:
  exit(0)

