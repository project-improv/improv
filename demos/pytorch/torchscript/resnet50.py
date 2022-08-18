import os
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights, progress=True).eval().to(self.device)

        self.classifier = nn.Linear(1000, 10)

        self.transform = weights.transforms()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = x.to(self.device)
            x = self.transform(x)
            features = self.model(x)
            predictions = self.classifier(features)

            return features, predictions
            
os.makedirs("models", exist_ok=True)
cpu_model = torch.jit.script(ResNet50('cpu')).save("models/ResNet50_CPU.pt")
gpu_model = torch.jit.script(ResNet50('cuda')).save("models/ResNet50_GPU.pt")