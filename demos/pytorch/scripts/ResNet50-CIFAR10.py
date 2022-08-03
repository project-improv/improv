import os
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights, progress=True)
    model.eval().to(device)

    transforms = torch.jit.script(weights.transforms()).to(device)

    ex = torch.rand(1, 3, 224, 224).to(device)
    
    traced_model = torch.jit.trace(model, ex).to(device)

    # Save the TorchScript model
    os.makedirs("/home/eao21/improv/demos/pytorch/models", exist_ok=True)
    transforms.save("/home/eao21/improv/demos/pytorch/models/ResNet50-transforms.pt")
    traced_model.save("/home/eao21/improv/demos/pytorch/models/ResNet50-CIFAR10.pt")

if __name__ == "__main__":
    main()