#!/bin/python
#=========================================================================
# pytorch-mnist
#=========================================================================

import torch
from torch                import nn
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.datasets import MNIST

#-------------------------------------------------------------------------
# Prepare Dataset
#-------------------------------------------------------------------------

train_data = MNIST( './data', train=True, download=True,
                    transform=transforms.ToTensor() )
test_data  = MNIST( './data', train=False, download=True,
                    transform=transforms.ToTensor() )

train_loader = DataLoader(train_data, batch_size=20, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=20, num_workers=0)

#-------------------------------------------------------------------------
# Multilayer Preception for MNIST
#-------------------------------------------------------------------------

class MLPModel(nn.Module):

  def __init__(self):
    super(MLPModel, self).__init__()

    self.mnist = nn.Sequential \
    (
      nn.Linear(784, 128),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 10),
    )

  def forward(self, x):
    return self.mnist(x)

#-------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------

model = MLPModel().cpu()
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


model.train()

print('Training starting ...')

for epoch in range(30):
  train_loss = 0.0

  for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data.view(-1, 28*28))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*data.size(0)

  train_loss = train_loss/len(train_loader.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f}'.format(
    epoch+1,
    train_loss
  ))
