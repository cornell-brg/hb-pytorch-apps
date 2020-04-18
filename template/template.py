"""
This is a workload tempalte for PyTorch-Apps repo.
Your workload should have the same structure.
04/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

#-------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------

"""
args = parse_model_args()
"""

#-------------------------------------------------------------------------
# Dataset Preprocessing
#-------------------------------------------------------------------------

"""
train_data = MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor())
test_data  = MNIST('./data', train=False, download=True,
                   transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)
"""

#-------------------------------------------------------------------------
# Model Name Here
#-------------------------------------------------------------------------

"""
class MLPModel(nn.Module):

    def __init__(self):
        super(MLPModel, self).__init__()

        self.model = nn.Sequential \
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
        return self.model(x)
"""

#-------------------------------------------------------------------------
# Model creation and loading
#-------------------------------------------------------------------------

"""
model = MLPModel()

# Load pretrained model if necessary
if args.load_model:
    model.load_state_dict( torch.load(args.model_filename) )

# Move model to HammerBlade if using HB
if args.hammerblade:
    model.to( torch.device("hammerblade") )

print( model )

# Quit here if dry run
if args.dry:
    exit( 0 )
"""

#-------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------

"""
if args.training:

    optimizer = torch.optim.SGD( model.parameters(), lr=0.01 )
    criterion = nn.CrossEntropyLoss()

    train \
        (
           model,
           train_loader,
           optimizer,
           criterion,
           args
         )
"""

#-------------------------------------------------------------------------
# Inference
#-------------------------------------------------------------------------

"""
if args.inference:

    criterion = nn.CrossEntropyLoss()

    inference \
            (
              model,
              test_loader,
              criterion,
              args
            )
"""

#-------------------------------------------------------------------------
# Model saving
#-------------------------------------------------------------------------

"""
if args.save_model:
    save_model( model, args.model_filename )
"""
