"""
pytorch-recsys
adapted from darpa-sdh-prog-eval/recsys
download data from: http://files.grouplens.org/datasets/movielens/ml-10m.zip
04/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from utils import parse_model_args, train, inference, save_model

# -------------------------------------------------------------------------
# Phase2 evaluation related commands
# -------------------------------------------------------------------------
import json
torch.hammerblade.init()

# Training phase chart
"""
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::div(const at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::embedding(const at::Tensor&, const at::Tensor&, int64_t, bool, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::mm(const at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::mul(const at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::relu(const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::view(const at::Tensor&, c10::IntArrayRef)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::batch_norm(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, double, double, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::binary_cross_entropy_with_logits(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::binary_cross_entropy_with_logits_backward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::dropout(const at::Tensor&, double, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::embedding_backward(const at::Tensor&, const at::Tensor&, int64_t, int64_t, bool, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::expand(const at::Tensor&, c10::IntArrayRef, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::ones_like(const at::Tensor&, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::sqrt(const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::sum(const at::Tensor&, c10::IntArrayRef, bool, c10::optional<c10::ScalarType>)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::t(const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::threshold_backward(const at::Tensor&, const at::Tensor&, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::unsqueeze(const at::Tensor&, int64_t)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::zeros_like(const at::Tensor&, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>)")
torch.hammerblade.profiler.chart.add("at::Tensor& at::CPUType::{anonymous}::add_(at::Tensor&, const at::Tensor&, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor& at::CPUType::{anonymous}::mul_(at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor& at::TypeDefault::addcdiv_(at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor& at::TypeDefault::addcmul_(at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar)")
torch.hammerblade.profiler.chart.add("int64_t at::TypeDefault::size(const at::Tensor&, int64_t)")
torch.hammerblade.profiler.chart.add("std::tuple<at::Tensor, at::Tensor, at::Tensor> at::CPUType::{anonymous}::native_batch_norm_backward(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, double, std::array<bool, 3>)")
"""

# Inference phase chart
"""
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::embedding(const at::Tensor&, const at::Tensor&, int64_t, bool, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::relu(const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::batch_norm(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, double, double, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::dropout(const at::Tensor&, double, bool)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::sum(const at::Tensor&, c10::IntArrayRef, bool, c10::optional<c10::ScalarType>)")
torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::t(const at::Tensor&)")
"""

# Training re-dispatch
"""
with open('training.json',) as f:
  route = json.load(f)
  torch.hammerblade.profiler.route.set_route_from_json(route)
"""

# Inference re-dispatch
"""
with open('inference.json',) as f:
  route = json.load(f)
  torch.hammerblade.profiler.route.set_route_from_json(route)
"""


# -------------------------------------------------------------------------
# Parse command line arguments
# -------------------------------------------------------------------------


def recsys_arg(parser):
    parser.add_argument('--dataset-path', type=str,
                        default='data/ml-10m.ratings.dat',
                        help="path to raw movielens dataset")
    parser.add_argument('--cache-path', type=str, default='data/cache',
                        help="path to preprocessed data cache")
    parser.add_argument("--validate", default=False, action='store_true',
                        help="run validation")


args = parse_model_args(recsys_arg)

# -------------------------------------------------------------------------
# Prepare Dataset
# -------------------------------------------------------------------------

# Check if cache already exists
if os.path.exists(args.cache_path + '_train.npy') \
        and os.path.exists(args.cache_path + '_valid.npy'):
    print('Cache exists ... skip dataset preprocessing')
else:
    print('loading %s' % args.dataset_path)
    # IO
    edges = pd.read_csv(args.dataset_path, header=None,
                        sep='::', engine='python')[[0, 1]]
    edges.columns = ('userId', 'movieId')

    # Remap IDs to sequential integers
    uusers = set(edges.userId)
    user_lookup = dict(zip(uusers, range(len(uusers))))
    edges.userId = edges.userId.apply(user_lookup.get)

    umovies = set(edges.movieId)
    movie_lookup = dict(zip(umovies, range(len(umovies))))
    edges.movieId = edges.movieId.apply(movie_lookup.get)
    edges.movieId += 1  # Add padding character

    # Train/test split
    train_data, valid_data = \
        train_test_split(edges, train_size=0.8, stratify=edges.userId)

    # Convert to adjacency list + save
    train_adjlist = \
        train_data.groupby('userId').movieId.apply(
            lambda x: sorted(set(x))).values
    valid_adjlist = \
        valid_data.groupby('userId').movieId.apply(
            lambda x: sorted(set(x))).values

    print('saving %s' % args.cache_path)
    np.save(args.cache_path + '_train', train_adjlist)
    np.save(args.cache_path + '_valid', valid_adjlist)


# Data loaders
class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks):
        self.X = [torch.LongTensor(xx) for xx in X]
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
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data
         for xx in X]

    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y


def make_dataloader(X, n_toks, batch_size, shuffle):
    return DataLoader(
        dataset=RaggedAutoencoderDataset(X=X, n_toks=n_toks),
        batch_size=batch_size,
        collate_fn=ragged_collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
    )


# Load data
X_train = np.load('%s_train.npy' % args.cache_path, allow_pickle=True)
n_toks = np.hstack(X_train).max() + 1
shuf_dataloader = make_dataloader(X_train, n_toks, 256, shuffle=True)
seq_dataloader = make_dataloader(X_train, n_toks, 256, shuffle=False)

# -------------------------------------------------------------------------
# Autoencoder based Recsys
# -------------------------------------------------------------------------


class MLPEncoder(nn.Module):
    def __init__(self, n_toks):
        super().__init__()

        self.emb = nn.Embedding(n_toks, 800, padding_idx=0)
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0

        self.act_bn_drop_1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(800),
            nn.Dropout(0.5),
        )

        self.bottleneck = nn.Linear(800, 400)
        self.bottleneck.bias.data.zero_()

        self.act_bn_drop_2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.5),
        )

        self.output = nn.Linear(400, n_toks)
        self.output.bias.data.zero_()
        self.output.bias.data += -10

    def forward(self, x):
        x = self.emb(x).sum(dim=1)
        x = self.act_bn_drop_1(x)
        x = self.bottleneck(x)
        x = self.act_bn_drop_2(x)
        x = self.output(x)
        return x


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _overlap(x, y):
    return len(set(x).intersection(y))


def compute_topk(X_train, preds, ks=[1, 5, 10]):
    max_k = max(ks)

    # --
    # Filter training samples

    # !! The model will tend to predict samples that are in the training data
    # and so (by construction) not in the validation data.  We don't want to
    # count these as incorrect though, so we filter them from the predictions
    low_score = preds.min() - 1
    for i, xx in enumerate(X_train):
        preds[i][xx] = low_score

    # --
    # Get top-k predictions

    # identical to `np.argsort(-pred, axis=-1)[:,:k]`, but should be faster
    topk = np.argpartition(-preds, kth=max_k, axis=-1)[:, :max_k]
    topk = np.vstack([r[np.argsort(-p[r])] for r, p in zip(topk, preds)])

    return topk


def compute_scores(topk, X_valid, ks=[1, 5, 10]):
    # --
    # Compute precision-at-k for each value of k

    precision_at_k = {}
    for k in ks:
        ps = [_overlap(X_valid[i], topk[i][:k]) for i in range(len(X_valid))]
        precision_at_k[k] = np.mean(ps) / k

    return precision_at_k


# -------------------------------------------------------------------------
# Model creation and loading
# -------------------------------------------------------------------------

model = MLPEncoder(n_toks=n_toks)

# Load pretrained model if necessary
if args.load_model:
    model.load_state_dict(torch.load(args.model_filename))

# Move model to HammerBlade if using HB
if args.hammerblade:
    model.to(torch.device("hammerblade"))

print(model)

# Quit here if dry run
if args.dry:
    exit(0)

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

if args.training:

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = F.binary_cross_entropy_with_logits

    train(model,
          shuf_dataloader,
          opt,
          loss,
          args)

# -------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------

if args.inference:

    loss = F.binary_cross_entropy_with_logits
    preds = []

    def collector(outputs, targets):
        preds.append(outputs.detach().cpu().numpy())

    inference(model,
              seq_dataloader,
              loss,
              collector,
              args)

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    if args.validate:
        preds = np.vstack(preds)

        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(X_train)
        assert preds.shape[1] == n_toks

        X_valid = np.load('%s_valid.npy' % args.cache_path, allow_pickle=True)
        topk = compute_topk(X_train, preds)

        scores = compute_scores(topk, X_valid)

        # --
        # Log

        P_AT_01_THRESHOLD = 0.475

        does_pass = "PASS" if scores[1] >= P_AT_01_THRESHOLD else "FAIL"
        print("Validation: %s with scores[1] = %f" % (does_pass, scores[1]))

# -------------------------------------------------------------------------
# Model saving
# -------------------------------------------------------------------------

if args.save_model:
    save_model(model, args.model_filename)
