import os
import torch
from resnet50 import ResNet50

device = torch.device("cpu")
                
model = ResNet50().to(device)
scripted_model = torch.jit.script(model).to(device)

os.makedirs("./models", exist_ok=True)
scripted_model.save("./models/ResNet50-CPU.pt")