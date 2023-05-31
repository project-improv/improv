#  https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
# tar -xvzf cifar-10-python.tar.gz
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/04_Retraining_Current_Architectures/04_download_cifar10.py
# https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
import os
import numpy as np
from skimage.io import imsave
import tarfile
import urllib.request


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


cifar_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

data_dir = "data/CIFAR10"
os.makedirs(data_dir, exist_ok=True)

cifar_file = os.path.join(data_dir, "cifar-10-python.tar.gz")
if not os.path.isfile(cifar_file):
    filename, headers = urllib.request.urlretrieve(cifar_link, cifar_file)

with tarfile.open(cifar_file) as tar:
    tar.extractall(path=data_dir)
    tar.close()

batch_path = os.path.join(data_dir, "cifar-10-batches-py")
batch_names = ["data_batch_" + str(x) for x in range(1, 6)]

file = os.path.join(batch_path, batch_names[0])

data_batch_1 = unpickle(file)
data = data_batch_1["data"]

labels = data_batch_1["labels"]

meta_file = os.path.join(batch_path, "batches.meta")
meta_data = unpickle(meta_file)

label_names = meta_data["label_names"]

os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "labels"), exist_ok=True)

for i in range(300):
    img = data[i]
    R = img[0:1024].reshape(32, 32)
    G = img[1024:2048].reshape(32, 32)
    B = img[2048:3072].reshape(32, 32)
    img = np.dstack((R, G, B))

    imsave(os.path.join(data_dir, "images/{}.jpg".format(i)), img)

    with open(os.path.join(data_dir, "labels/{}.txt".format(i)), "w") as text_file:
        text_file.write("%s" % labels[i])
        text_file.close()

with open(os.path.join(data_dir, "label_names.txt"), "w") as text_file:
    text_file.write(", ".join(label_names))
    text_file.close()
