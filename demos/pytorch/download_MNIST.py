import os
from torchvision import datasets

def main():

    os.makedirs('./data', exist_ok=True)
    os.chdir('./data')
    
    datasets.MNIST('./')

if __name__ == "__main__":
    main()