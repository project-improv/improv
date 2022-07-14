import os
import shutil
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

train_percent = 0.8

def main():

    # Tmp dir in $HOME
    os.makedirs("data", exist_ok=True)
    
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Random sample for smaller dataset
    train_size = int(len(train_data) * train_percent)
    test_size = len(train_data) - train_size
    train_subset, test_subset = random_split(train_data, [train_size, test_size])

    # Create DataLoader?
    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl

if __name__ == "__main__":
    main()