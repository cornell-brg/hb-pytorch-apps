import argparse
import torch
import numpy as np

ATOL = 1e-5
RTOL = 1e-5

def is_model_on_hammerblade(model):
    return next(model.parameters()).is_hammerblade

def argparse_inference():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nbatch', default=1, type=int,
                        help="Number of batches to be tested")
    parser.add_argument('--hammerblade', default=False, action='store_true',
                        help="Run on HammerBlade")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Training batch size")
    parser.add_argument("--dry", default=False, action='store_true',
                        help="Dry run")
    parser.add_argument('--filename', default="trained_model", type=str,
                        help="Filename of the saved model")
    return parser.parse_args()

def argparse_training():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nepoch', default=1, type=int,
                        help="Number of training epochs")
    parser.add_argument('--nbatch', default=None, type=int,
                        help="Number of batches to train on in each epoch." \
                             " Mainly useful to limit a training run to few batches."
                             " If None, runs on the whole dataset.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Training batch size")
    parser.add_argument('--hammerblade', default=False, action='store_true',
                        help="Run on HammerBlade")
    parser.add_argument("--dry", default=False, action='store_true',
                        help="Dry run")
    parser.add_argument("--save-model", default=False, action='store_true',
                        help="Save trained model to file")
    parser.add_argument('--save-filename', default="trained_model", type=str,
                        help="Filename of the saved model")
    return parser.parse_args()

# Train routine
def train(model, loader, optimizer, loss_func, epochs, batches=None):
    print('Training {} for {} epoch(s)...\n'.format(type(model).__name__, epochs))
    for epoch in range(epochs):
        losses = []

        for batch_idx, (data, labels) in enumerate(loader, 0):
            if is_model_on_hammerblade(model):
                data, labels = data.hammerblade(), labels.hammerblade()
            batch_size = len(data)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (batches is None and batch_idx % 1000 == 0) or \
                    (batches is not None and batch_idx < batches):
                print('epoch {} : [{}/{} ({:.0f}%)]\tLoss={:.6f}'.format(
                    epoch, (batch_idx + 1) * batch_size, len(loader.dataset),
                    100. * (batch_idx / len(loader)), loss.item()
                ))
            else:
                break

        print('epoch {} : Average Loss={:.6f}\n'.format(
            epoch, np.mean(losses)
        ))

# Test routine
@torch.no_grad()
def test(model, loader, loss_func, nbatch):
    test_loss = 0.0
    num_correct = 0

    for batch_idx, (data, labels) in enumerate(loader, 0):
        if is_model_on_hammerblade(model):
            data, labels = data.hammerblade(), labels.hammerblade()
        output = model(data)
        loss = loss_func(output, labels)
        pred = output.max(1)[1]
        num_correct += pred.eq(labels.view_as(pred)).sum().item()

        if batch_idx == nbatch:
            break

    test_loss /= len(loader.dataset)
    test_accuracy = 100. * (num_correct / len(loader.dataset))

    print('Test set: Average loss={:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, num_correct, len(loader.dataset), test_accuracy
    ))
