import os
import torch
from torchvision import models
# Specifically for AlexNet --- could make generalizable, not necessary
from torchvision.models import AlexNet_Weights

# Default GPU=True or False...
# https://stackoverflow.com/questions/3577163/how-to-input-variable-into-a-python-script-while-opening-from-cmd-prompt
# If w/sys.argv:
# ''' \
# USAGE:  python download_AlexNet.py -i1 model_url -i2 model_name -i3 GPU -i4 example_size -i5 traced_model_name
# ETC.
# '''

def main():

    # Tmp dir in $HOME
    os.makedirs('./models', exist_ok=True)

    # Default model_dir is $TORCH_HOME/models, $TORCH_HOME defaults to ~/.torch
    os.environ['TORCH_HOME'] = './models/'
    
    # Download model from PyTorch
    # Arbitrary...from torchvision.models OR URL:
    # EX FROM URL:
    # "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    # wget -c https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth - NOT SURE WHICH VERSION, WHY DIFFERENT THAN ABOVE?
    # IN PYTHON VS COMMAND:
    # As user input for script in command line before running improv
    import sys
    # url = sys.argv[1]
    # url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    # from six.moves import urllib
    # os.chdir('./models)
    # model_name = sys.argv[2]
    # model_name = "alexnet.pth"
    # urllib.request.urlretrieve(url, model_name)
    weights = AlexNet_Weights.DEFAULTS
    model = models.alexnet(weights=weights, progress=True)
    model.eval()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing - might want to use script instead of trace?
    # Must use example data set of same size as input to model - either to GPU or CPU, could test both
    example_size = 
    example = torch.rand(example_size)
    traced_model = torch.jit.trace(model, example)
    # Save the TorchScript model
    traced_model.save('traced_alexnet.pt')
    
if __name__ == "__main__":
    main()