import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
model.to(torch.device("hammerblade"))

print(model(torch.rand(1, 3, 32, 32)))
