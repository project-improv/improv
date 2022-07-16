import os
from torchvision import datasets
# Specifically for MNIST and AlexNet and transform for AlexNet --- could make generalizable, not necessary rn...
from torchvision.models import AlexNet_Weights

def main():
# Or:
# import sys
# sys.argv... etc.

    # Make dir in curr dir =  ~/improv/demos/pytorch/
    # os.makedirs('./data/MNIST-AlexNet', exist_ok=True)

    # Transform for AlexNet - how does input need to be transformed for models? How long will this take?
    # ToTensor for VAE?
    transform = AlexNet_Weights.DEFAULT.transforms
    
    # Download training data from open datasets to /data/MNIST folder
    data = datasets.MNIST(root=os.getcwd, download=True, transform=transform)

    # Not necessary to check from store -> model -> store, only inference
    # Download test data from open datasets to /data/MNIST folder
    # test_data = datasets.MNIST(root='./data/MNIST', download=True, transform=transform, train=False)

    # Above will save data out for store, as if we already have data...will sample one "datapoint"/sample at a time for inference and time/sample

    # NOT NECESSARY TO CREATE DATALOADER - SLOW, see implementation
    # Create DataLoader
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
if __name__ == "__main__":
    main()