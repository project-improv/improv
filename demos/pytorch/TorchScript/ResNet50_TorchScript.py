import os
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

gpu_num = 0

class RN(nn.Module):
    
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights, progress=True).eval()
        self.transform = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transform(x)
            return self.model(x)

device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
                
model = RN().to(device)
scripted_model = torch.jit.script(model).to(device)

os.makedirs("./models", exist_ok=True)
scripted_model.save("./models/scripted_ResNet50.pt")