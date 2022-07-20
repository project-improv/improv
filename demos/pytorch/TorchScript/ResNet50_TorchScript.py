import os
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor
# from torch.utils.data import ToTensor

def main():
    
    class RN(nn.Module):
        
        def __init__(self, weights=ResNet50_Weights.DEFAULT, model=resnet50):
            super(RN, self).__init__()
            # For MNIST:
            # self.param = nn.Parameter(torch.rand(28, 28))
            self.weights = weights
            self.model = model(weights=self.weights, progress=True).eval()
            self.transforms = weights.transforms()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                x = self.transforms(x)
                return self.model(x)

    model = RN()
    scripted_model = torch.jit.script(model)
    # model_fx = create_feature_extractor(model, return_nodes=['add'])
    # scripted_model = torch.jit.script(model_fx).to(device)

    # For MNIST data example size, one "batch", one image to simulate model seeing one input at a time:
    example_size = (1, 1, 28, 28)
    # For CIFAR-10 data example size, one "batch", one image to simulate model seeing one input at a time:
    # example_size = (1, 3, 32, 32)
    # Without hard-coding example size...
    # dataset = datasets.MNIST(root="./data", download=False, transform=transforms.ToTensor())

    example = torch.rand(example_size)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
    traced_model = torch.jit.trace(scripted_model, example)

    # Save the TorchScript model
    os.makedirs("./models", exist_ok=True)
    traced_model.save("./models/scripted_ResNet50.pt")

if __name__ == "__main__":
    main()