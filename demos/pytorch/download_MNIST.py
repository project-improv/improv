import os
from torchvision import datasets

# Run in ~/improv/demos/pytorch
def main():

    os.makedirs("./data", exist_ok=True)
    
    datasets.MNIST(root="./data", download=True)

if __name__ == "__main__":
    main()