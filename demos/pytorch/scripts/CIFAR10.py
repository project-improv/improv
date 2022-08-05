import os
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Run in ~/improv/demos/pytorch
def main():

    os.makedirs("data/CIFAR10/images", exist_ok=True)
    os.makedirs("data/CIFAR10/labels", exist_ok=True)

    batch_size = 200
    
    dataset = datasets.CIFAR10(root="data", download=True, transform=ToTensor())
    dataset = iter(DataLoader(dataset, batch_size=batch_size))
    
    for i in range(batch_size):
        x, label = (next(dataset))
        save_image(x[0], "data/CIFAR10/images/{}.jpg".format(i))
        with open("data/CIFAR10/labels/{}.txt".format(i), "w") as text_file:
            text_file.write("%s" % label)

if __name__ == "__main__":
    main()