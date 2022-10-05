# Convolutional Neural Network (CNN) Demo

Demonstration of image feature extraction and classification using a CNN within improv.

(cnn_demo.py and cnn_demo.yaml)

(For timing: cnn_demo_timing.py and cnn_demo_timing.yaml)

## Installation

### Dependencies (in addition to improv dependencies):
* [PyTorch](https://pytorch.org)

#### CUDA Toolkit
**NOTE:** To use a GPU, the CUDA Toolkit must be installed.

(Specific for lab computer...for Chris)
First, install the NVCC driver following the [CUDA 11.6.0 Toolkit Documentation for Linux](https://docs.nvidia.com/cuda/archive/11.6.0/cuda-installation-guide-linux/index.html):

**SEE SECTION 2 and walkthrough, skip 2.5**

Type 'nvidia-smi' to verify installation of the NVCC driver. You should see GPU information.

#### PyTorch
Follow the instructions on the PyTorch homepage for installation, i.e., type the following after activating your conda env:

'conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge'

This is the most basic, simplest install. (This actually did not work for me, which is totally fine...there are so many other ways to install!)

If this does not work, contact me ASAP, and I will send more in-depth install instructions based on your specific errors.

## Data — CIFAR10 Images 

(scripts/CIFAR10.py)

See the following documentation for information on the CIFAR-10 and CIFAR-100 datasets:

[CIFAR Dataset Information](https://www.cs.toronto.edu/~kriz/cifar.html)

To create data (one batch of images (10000 images/batch) from the CIFAR-10 dataset saved as .jpg files), type the following in the base dir, i.e., cnn folder:

'python scripts/CIFAR10.py'

In this script:
1. Data is downloaded from a URL as a .tar.gz file and extracted into memory.
2. A single batch of images and meta data are unpickled
3. A single batch of images, labels for each image, and a list of all possible label names are saved as .jpg, .txt, and .txt files, respectively.

**NOTE:** You can change the number of images to any value greater than 0 and less than or equal to 10000 (there are 10000 images/batch) or use more than one batch.

This script creates the following:
* data/CIFAR10
    * cifar-10-python.tar.gz: CIFAR-10 datasets as .tar.gz file
    * label_names.txt: list of label names as a .txt file
    * cifar-10-batches-py: folder of pickled batched data
    * images: one batch, 10000, CIFAR-10 images as .jpg files
    * labels: one batch, 10000, of corresponding labels (ints) as .txt files

## Model — Pretrained ResNet50 with Default Weights (trained on the ImageNet dataset)

(torchscript/resnet50.py)

See the following documentation for more information on the ResNet50 model:

[ResNet50 PyTorch Documentation](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/) and [ResNet50 Torchvision Documentation](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

To create the model, type the following in the base dir, i.e., cnn folder:

'python torchscript/resnet50.py'

In this script:
1. The ResNet50 model is constructed using default weights (SOTA).
2. Transforms are added to the forward pass.
3. A final linear classification layer is added.
4. CPU and GPU models are saved.

**NOTE:** Models are compiled as JIT scripts.

**NOTE:** Any pretrained CNN with any weights can be swapped out (line 11-12), and the dimensionality of the final fully connected linear classification layer can be customized or multiple layers can be added. The final dimension must be 10 layers or you can also add a softmax function to give 1 layer. Otherwise, the softmax is incorporated into the function to predict labels.

This script creates the following:
* models
    * ResNet50_CPU.pt
    * ResNet50_GPU.pt

The output of the models is both features and predictions.

## Acquisition

(actors/acquire.py)

Simulating acquisition of .jpg images and .txt labels.

ImageAcquirer loops over all image files in a folder containing images as either .jpg or .png files. If running classification, it also loops over all label files in a folder containing labels as .txt files.

Images (and labels, if running classification) are put out to store and Processor.

## Processing

Processing images includes preprocessing images, running inference to get features and predictions, and classifying images.

CNNProcessor does the following:
1. Loads a CNN model.
2. Warms up the model (the number of runs can be changed).
3. Processes the image (input from Acquirer), i.e., converts an image from a numpy array to tensor, unsqueezes the tensor, and changes the ordering of the channels.
4. Inference is run (outputs = features+predictions, time to device, inference time).
5. Images are classified if running classification (outputs = predicted labels, percent confidence, and scores and associated labels for the top five predictions).
