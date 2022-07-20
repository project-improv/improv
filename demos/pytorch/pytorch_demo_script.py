import torch

from demos.pytorch.actors.pytorch_processor import PyTorchProcessor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    '''
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pt_p = PyTorchProcessor("PyTorch", data_path='~/improv/demos/pytorch/data/CIFAR10', model_path='~/improv/demos/pytorch/models/AlexNet-CIFAR10.pt', config_file='~/improv/demos/pytorch/pytorch_demo.yaml', device=device)

    print(pt_p)
    
if __name__ == "__main__":
    main()






        

