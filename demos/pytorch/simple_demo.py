# 1. Download data
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

# Download MNIST training data from torchvision open datasets
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download MNIST test data from torchvision open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Subset each for smaller dataset
split_len = 1000

subset_train = random_split(train_data, [split_len, len(train_data)-split_len])[0]

subset_train = random_split(test_data, [split_len, len(test_data)-split_len])[0]

# *time*

# 2. Start store

# 3. Put data in store

# 4. Define model (maybe above?)

# 5. Take data out of store

# 6. Load data into model

# 7. Run model

# 8. Viz