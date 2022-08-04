import os
import torch
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import datasets
# from torch.utils.data import ToTensor

class AN(nn.Module):

    def __init__(self, weights=AlexNet_Weights.DEFAULT, model=alexnet):
        super(AN, self).__init__()
        self.weights = weights
        self.model = alexnet(weights=self.weights, progress=True).eval()
        self.transforms = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            return self.model(x)

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # weights = AlexNet_Weights.DEFAULT
    # model = alexnet(weights=weights, progress=True).eval()
    model = AN().to(device)
    # model = AN()

    scripted_model = torch.jit.script(model).to(device)

    # For MNIST data example size, one "batch", one image to simulate model seeing one input at a time:
    # example_size = (1, 3, 32, 32)
    # example_size = (28, 28)
    # Without hard-coding example size...
    # dataset = datasets.MNIST(root="./data", download=False, transform=transforms.ToTensor())

    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
    torchscript_AN = torch.jit.trace(model, (example))

    # Save the TorchScript model
    os.makedirs("./models", exist_ok=True)
    torchscript_AN.save("./models/traced_AlexNet.pt")

if __name__ == "__main__":
    main()