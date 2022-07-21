def main():
    '''
    '''

    import sys # issue w/PYTHONPATH and sys.path...improv src not in there...modules not loading...

    sys.path.append('/home/eao21/improv')

    import logging
    # Matplotlib is overly verbose by default
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # import subprocess

    from improv.nexus import Nexus

    import torch

    from demos.pytorch.actors.pytorch_processor import PyTorchProcessor

    import logging; logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    loadFile = '/demos/pytorch/pytorch_demo.yaml'

    # subprocess.Popen(
    # ["plasma_store", "-s", "/tmp/store", "-m", str(10000000)],\
    # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    nexus = Nexus('Nexus')
    nexus.createNexus(file=loadFile)

    # All modules needed have been imported
    # so we can change the level of logging here
    # import logging
    # import logging.config
    # logging.config.dictConfig({
    #     'version': 1,
    #     'disable_existing_loggers': True,
    # })
    # logger = logging.getLogger("improv")
    # logger.setLevel(logging.INFO)

    nexus.startNexus()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pt_p = PyTorchProcessor('PyTorch', data_path='/home/eao21/improv/demos/pytorch/data/CIFAR10', model_path='/home/eao21/improv/demos/pytorch/models/AlexNet-CIFAR10.pt', config_file='/home/eao21/improv/demos/pytorch/pytorch_demo.yaml', device=device)
    
if __name__ == "__main__":
    main()






        

