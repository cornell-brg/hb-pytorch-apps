from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch

import unicodedata
import string
import random 

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        
         return torch.zeros(1, self.hidden_size, device="hammerblade")
#         return torch.zeros(1, self.hidden_size)



def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)      
    return output, loss.item()



def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor = torch.zeros(1, n_letters, device="hammerblade")
#    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters, device="hammerblade")
#    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
#    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long, device="hammerblade")
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def categoryFromOutput(output):
    output = output.cpu()
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

if __name__ == "__main__":

  category_lines = {}
  all_categories = []
  all_letters = string.ascii_letters + " .,;'"
  n_letters = len(all_letters)
  
  for filename in findFiles('data/data_simple_rnn/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
 

  n_categories = len(all_categories)
  

  n_hidden = 128
  print("input size: " + str(n_letters))
  print("output size: " + str(n_categories))
  rnn = RNN(n_letters, n_hidden, n_categories)
  rnn.to(device="hammerblade")
  criterion = nn.NLLLoss() 
  learning_rate = 0.005
 
  n_iters = 100000

# Keep track of losses for plotting
  current_loss = 0

  torch.hammerblade.profiler.enable()

  print("training")
  for iter in range(1, 2):
    #  print(iter)
      category, line, category_tensor, line_tensor = randomTrainingExample()
      output, loss = train(category_tensor, line_tensor)
      current_loss += loss
      
#  confusion = torch.zeros(n_categories, n_categories)
#  n_confusion = 10000

#  correct = 0
  # Go through a bunch of examples and record which are correctly guessed
#  for i in range(n_confusion):
#      category, line, category_tensor, line_tensor = randomTrainingExample()
#      output = evaluate(line_tensor)
#      guess, guess_i = categoryFromOutput(output)
#      category_i = all_categories.index(category)

#      if category_i == guess_i:
#         correct += 1
#      confusion[category_i][guess_i] += 1
  
#  print(correct/n_confusion)
# Normalize by dividing every row by its sum
#  for i in range(n_categories):
#      confusion[i] = confusion[i] / confusion[i].sum()



  torch.hammerblade.profiler.disable()
  print(torch.hammerblade.profiler.exec_time.raw_stack())
  print(torch.hammerblade.profiler.exec_time.fancy_print())

  
