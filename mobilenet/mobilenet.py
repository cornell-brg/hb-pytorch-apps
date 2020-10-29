import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)

model_hb = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
model_hb.to(torch.device("hammerblade"))

out = model(torch.rand(1, 3, 224, 224))
out_hb = model_hb(torch.rand(1, 3, 224, 224).hammerblade())

assert torch.allclose(out, out_hb.cpu())
