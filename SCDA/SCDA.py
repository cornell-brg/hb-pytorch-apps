import argparse
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from torch import nn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# specify a seed for repeating the exact dataset splits
torch.manual_seed(28213)
np.random.seed(seed=28213)

# hyperparameters
lr          = 0.0001
max_epochs  = 10
filter_size = 5
use_bias    = True
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 0}

parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--hammerblade', default=False, action='store_true',
                            help="run SCDA on HammerBlade")

#--------------------------------------------------------------#
# Sparse Convolutional Denoising Autoencoder                   #
# ported from https://github.com/work-hard-play-harder/SCDA    #
#--------------------------------------------------------------#

class SCDA(nn.Module):

  def __init__(self, in_channels):
    super(SCDA, self).__init__()

    # encoder
    self.encoder = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=filter_size, bias=use_bias, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(0.25),
        nn.Conv1d(32, 64, kernel_size=filter_size, bias=use_bias, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(0.25)
    )

    # bridge
    self.bridge = nn.Conv1d(64, 128, kernel_size=filter_size, bias=use_bias, padding=2)

    # decoder
    self.decoder = nn.Sequential(
        nn.Conv1d(128, 64, kernel_size=filter_size, bias=use_bias, padding=2),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Dropout(0.25),
        nn.Conv1d(64, 32, kernel_size=filter_size, bias=use_bias, padding=2),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Dropout(0.25),
        nn.Conv1d(32, in_channels, kernel_size=filter_size, bias=use_bias, padding=2),
    )


  def forward(self, x):
    x = self.encoder(x)
    x = self.bridge(x)
    x = self.decoder(x)
    return x

#-------------------------------------------------------------
# Helpers
#-------------------------------------------------------------

def df_to_tensor(df):
  return torch.from_numpy(df.values)

class Dataset(torch.utils.data.Dataset):

  def __init__(self, x, y, missing_perc=0.1):
    self.x = x
    self.y = y
    self.missing_perc = missing_perc

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    item = self.y[index]
    miss = torch.clone(self.x[index])
    missing_size = int(self.missing_perc * miss.size(-1))
    missing_index = np.random.randint(miss.size(-1), size=missing_size)
    for i in missing_index:
      miss[0][i] = 1.0
      for c in range(1,miss.size(0)):
        miss[c][i] = 0.0
    return miss, item


#-------------------------------------------------------------
# Main
#-------------------------------------------------------------

if __name__ == "__main__":
  # parse args
  args = parser.parse_args()

  # test with yeast data
  input_name = '/work/global/lc873/work/bio/SCDA/data/yeast_genotype_train.txt'
  df_ori = pd.read_csv(input_name, sep='\t', index_col=0)
  # make it fit
  df_ori = df_ori.iloc[:,:28160]
  print("yeast data loaded. Shape = " + str(df_ori.shape))
  print("sample data:")
  print(df_ori.head())

  # train / test split
  train_ori, valid_ori = train_test_split(df_ori, test_size=0.2)

  tensor_train  = df_to_tensor(train_ori)
  tensor_valid  = df_to_tensor(valid_ori)
  train_one_hot = nn.functional.one_hot(tensor_train).float()
  valid_one_hot = nn.functional.one_hot(tensor_valid).float()
  print("train set shape - " + str(train_one_hot.size()))
  print("valid set shape - " + str(valid_one_hot.size()))

  # init SCDA model
  model = SCDA(train_one_hot.size(-1))
  if args.hammerblade:
    model.to(torch.device("hammerblade"))
  print()
  print(model)
  print()

  # convert channel last to channel first
  print("channel last one hot tensor: " + str(train_one_hot.size()))
  print()
  train_one_hot = train_one_hot.permute(0,2,1).contiguous()
  valid_one_hot = valid_one_hot.permute(0,2,1).contiguous()
  print("channel first one hot tensor: " + str(train_one_hot.size()))

  # create data loaders
  training_set   = Dataset(train_one_hot, tensor_train)
  validation_set = Dataset(valid_one_hot, tensor_valid)
  training_generator   = torch.utils.data.DataLoader(training_set, **params)
  validation_generator = torch.utils.data.DataLoader(validation_set, **params)

  # train the model
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(max_epochs):
    model.train()
    losses = []
    for batch_idx, (x, y) in tqdm(enumerate(training_generator, 0), total=len(training_generator)):
      x, y = x.contiguous(), y.contiguous()
      if args.hammerblade:
        x, y = x.hammerblade(), y.hammerblade()
      optimizer.zero_grad()
      z = model(x)
      loss = criterion(z,y)
      losses.append(loss.item())
      loss.backward()
      optimizer.step()

    print('epoch {} : Average Training Loss={:.6f}\n'.format(
      epoch,
      np.mean(losses)
    ))

    # validation
    model.eval()
    losses = []
    for batch_idx, (x, y) in tqdm(enumerate(validation_generator, 0), total=len(validation_generator)):
      x, y = x.contiguous(), y.contiguous()
      z = model(x)
      loss = criterion(z,y)
      losses.append(loss.item())
      loss.backward()
      optimizer.step()

    print('epoch {} : Average Validation Loss={:.6f}\n'.format(
      epoch,
      np.mean(losses)
    ))
