import os
import torch
from torchvision import models
# Specifically for AlexNet --- could make generalizable, not necessary
from torchvision.models import AlexNet_Weights

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
    # url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    # from six.moves import urllib
    # os.chdir('./models)
    # urllib.request.urlretrieve(url, "alexnet.pth")
    weights = AlexNet_Weights.DEFAULTS
    models.alexnet(weights=weights, progress=True)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing - might want to use script instead of trace?
    traced_model = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")
    
if __name__ == "__main__":
    main()