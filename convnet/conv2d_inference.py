import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, help="Number of inputs to be processed for conv2d.")
parser.add_argument('--channels', type=int, help="Number of input channels for conv2d.")
parser.add_argument('--filters', type=int, help="Number of kernel channels for conv2d.")
parser.add_argument('--image-dim', type=int, help="Size of inputs to be processed for conv2d. Square image.")
parser.add_argument('--kernel-size', type=int, help="Size of kernel for conv2d.")
parser.add_argument('--padding', action="store_ture")
args = parser.parse_args()

batch_size = args.batch_size
channels = args.channels
filters = args.filters
image_dim = args.image_dim
kernel_size = args.kernel_size

# Build inputs
imap = torch.tensor([[[[float(c+i+r+channel) for c in range(image_dim)] for r in range(image_dim)] for channel in range(channels)] for i in range(batch_size)], requires_grad=False)
hmap = imap.clone().hammerblade().detach().requires_grad_(True)
print("imaps:")
print(imap.size())

# Build filters
cweight = torch.tensor([[[[float(c*(k+1)+r+1+channel) for c in range(kernel_size)] for r in range(kernel_size)] for channel in range(channels)] for k in range(filters)], requires_grad=False)
hweight = cweight.clone().hammerblade().detach().requires_grad_(True)
print("filters:")
print(cweight.size())

bias = None
padding = (1,) if args.padding else (0,)
stride = (1,)
dilation = (1,)
transposed = False
output_padding = (0,)
groups = 1

y = torch.convolution(imap, cweight, bias, stride, padding, dilation, transposed, output_padding, groups)

print("cpu omaps:")
print(y.size())
print("++++----")

z = torch.convolution(hmap, hweight, bias, stride, padding, dilation, transposed, output_padding, groups)

print("hb omaps:")
print(z.size())
print("++++----")

assert torch.allclose(y, z.cpu(), rtol=0.0001)
