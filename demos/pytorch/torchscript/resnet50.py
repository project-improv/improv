import os
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights, progress=True).eval()
        self.transform = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transform(x)
            return self.model(x)