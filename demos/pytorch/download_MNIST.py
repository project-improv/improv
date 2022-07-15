import os
import torch
from torchvision import datasets
# Specifically for MNIST and AlexNet and tranform for AlexNet --- could make generalizable, not necessary
from torchvision.models import AlexNet_Weights

def main():
# def main(dataset)

    # Make dir in curr dir =  ~/improv/demos/pytorch/
    os.makedirs('./data/MNIST', exist_ok=True)

    # Transform for AlexNet - how does input need to be transformed for models? How long will this take?
    # ToTensor for VAE?
    transform = AlexNet_Weights.DEFAULT.transforms
    
    # Download training data from open datasets to /data/MNIST folder
    train_data = datasets.MNIST(root='./data/MNIST',
                               download=True,transform=tranform,
                               train=True)

    # Download test data from open datasets to /data/MNIST folder
    test_data = datasets.MNIST(root='./data/MNIST',
                              download=True,transform=transform,
                              train=False)

    # Above will save data out for store, as if we already have data...will sample one "datapoint"/sample at a time for inference and time/sample

    # NOT NECESSARY TO CREATE DATALOADER - SLOW, see implementation
    # Create DataLoader
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
if __name__ == "__main__":
    main()