
import os
import torch

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")  # or "resnet50"
model.cpu()
model.eval()

traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module.save("rvm_ts.pt")

os.system("./pnnx rvm_ts.pt inputshape=[1,3,512,512] device=cpu") 


