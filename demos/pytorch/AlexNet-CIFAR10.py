import os
import torch
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights, progress=True).eval().to(device)

    transforms = torch.jit.script(weights.transforms()).to(device)

    # Load image, tranform to tensor using torchvision.ToTensor
    # Single image - unsqueeze (1, 3, h, w)
    # For CIFAR-10 - dk why it doesn't work w/MNIST
    # MNIST - 1, 1, 28, 28
    example = torch.rand(1, 3, 32, 32).to(device)

    traced_model = torch.jit.trace(model, transforms(example)).to(device)

    # Save the TorchScript model
    os.makedirs("./models", exist_ok=True)
    traced_model.save("./models/AlexNet-CIFAR10.pt")


if __name__ == "__main__":
    main()