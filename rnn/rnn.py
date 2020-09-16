"""
RNN model used to predict number of "stars" (1-5) given a corresponding yelp review
Model uses pytorch nn.RNN 
helpful links: https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
https://pytorch.org/docs/stable/nn.html

Changing hidden dim will affect the accuracy and time to run
To run with different word embeds or other hyper params the code must be changed
04/23/2020 Jack Weber (jlw422@cornell.edu)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import json
from utils import parse_model_args, train, inference, save_model
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------------
# RNN for sentiment prediction
# -------------------------------------------------------------------------


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, output_size):
        super(RNN, self).__init__()


        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, 1, batch_first=True)

        self.i2o = nn.Linear(hidden_size, output_size)

# forward pass 
    def forward(self, input):
        #formats the input to correct dimensions for rnn
        input = torch.stack(input)
        input = input.unsqueeze(1)

        output, hidden = self.rnn(input, None)

        #dim manipulation
        output = output.contiguous().view(-1, self.hidden_size)

        #put output through linear layer to give proper dimension for task
        output = self.i2o(output)
        return output, hidden

# -------------------------------------------------------------------------
# Workload specific command line arguments
# -------------------------------------------------------------------------


def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.01, type=int,
                        help="learning rate")

    parser.add_argument('--hd', default=10, type=int,
                        help="hidden dimension")

    parser.add_argument('--ed', default=10, type=int,
                        help="embedding dimension")


def fetch_data():
    with open('training.json') as training_f:
        training = json.load(training_f)
    with open('validation.json') as valid_f:
        validation = json.load(valid_f)
    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val



# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # Parse command line arguments
    # ---------------------------------------------------------------------

    args = parse_model_args(extra_arg_parser)

    # ---------------------------------------------------------------------
    # Prepare Dataset
    # ---------------------------------------------------------------------

    train_data, valid_data = fetch_data()

    EMBEDDING_DIM = args.ed 

    vocab = set()

    # use to create embedding for unknown words during inference
    vocab.add("unk")
    for document, _ in train_data:
        for word in document:
            vocab.add(word)

    vocab_list = sorted(vocab)

    word2index = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 

    embeds = nn.Embedding(len(word2index), EMBEDDING_DIM)

    # ---------------------------------------------------------------------
    # Model creation and loading
    # ---------------------------------------------------------------------


    LEARNING_RATE = args.lr
    HIDDEN_DIM = args.hd

    model = RNN(EMBEDDING_DIM, hidden_size = HIDDEN_DIM, vocab_size= len(vocab), output_size = 5)

    # changing these parameters will effect accuracy of model
    optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum=0.9)

    # use CEL to compute loss for back prop
    criterion = nn.CrossEntropyLoss()


    # Load pretrained model if necessary
    if args.load_model:
        model.load_state_dict(torch.load(args.model_filename))

    # Move model to HammerBlade if using HB
    if args.hammerblade:
        model.to(torch.device("hammerblade"))
        embeds.to(torch.device("hammerblade"))

    print(model)

    # Quit here if dry run
    if args.dry:
        exit(0)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    if args.training:
        print("training")
        EPOCHS = args.nepoch


        random.shuffle(train_data)
        for epoch in range(EPOCHS):
            for sent, target in train_data:

                inp = []
                targets = []
                for w in sent:
                   inp.append(embeds(torch.tensor(word2index[w], dtype=torch.long, device="hammerblade")))

                output, hidden = model(inp)

                targets = torch.tensor([target for i in range(0,output.shape[0])], dtype=torch.long, device="hammeblade")

                # back prop loss
                loss = criterion(output, targets)
                model.zero_grad()
                loss.backward()
                optimizer.step()

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    if args.inference:
        print("started inference")
        y_pred = []
        y_actual = []
        model.train(False)
        with torch.no_grad():
            for sent in valid_data:
                sent_tag =sent[1]
                inp = []

                for w in sent[0]:
                    if w in vocab:
                        inp.append(embeds(torch.tensor(word2index[w], dtype=torch.long)))
                    else:
                        inp.append(embeds(torch.tensor(word2index["unk"], dtype=torch.long)))

                class_scores, hidden = model(inp)
                y_pred.append(str(class_scores.max(dim=1)[1].numpy()[0]))

                y_actual.append(str(sent_tag))
        print(accuracy_score(y_actual, y_pred))

    # ---------------------------------------------------------------------
    # Model saving
    # ---------------------------------------------------------------------

    if args.save_model:
        save_model(model, args.model_filename)
