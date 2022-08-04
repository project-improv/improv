import logging
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from improv.nexus import Nexus

loadFile = './pytorch_demo.yaml'

# import torch.multiprocessing as mp
# from multiprocessing import set_start_method, get_context

# mp.get_context("forkserver")
# mp.set_start_method("forkserver", force=True)

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