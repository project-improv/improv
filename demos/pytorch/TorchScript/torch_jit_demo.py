import torchvision
import torch
from time import perf_counter
import numpy as np

# Based on: https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff

def timer(f, *args):   
    start = perf_counter()
    # Necessary?
    f(*args)
    # Why this line? Change 1000
    return (1000 * (perf_counter() - start))

def main():
    # Example PyTorch CPU demo
    model= torchvision.models.resnet18(pretrained=True)
    model.eval()
    # Input size = 1, 3, 224, 224...fed in 10 times?
    x = torch.rand(1, 3, 224, 224)
    print(np.mean([timer(model, x) for _ in range(10)]))

    # Example PyTorch GPU demo
    model_gpu = torchvision.models.resnet18(pretrained=True).cuda()
    x_gpu = x.cuda()
    model_gpu.eval()
    print(np.mean([timer(model_gpu, x_gpu) for _ in range(10)]))

    # Example torch.jit.script CPU version
    script_cell = torch.jit.script(model, (x))
    print(np.mean([timer(script_cell, x) for _ in range(10)]))

    # Example torch.jit.script GPU version
    script_cell_gpu = torch.jit.script(model_gpu, (x_gpu))
    print(np.mean([timer(script_cell_gpu, x_gpu) for _ in range(10)]))

if __name__ == "__main__":
    main()