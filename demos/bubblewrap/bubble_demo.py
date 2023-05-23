import logging

# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from improv.nexus import Nexus

import os
from pathlib import Path
from multiprocessing import set_start_method, get_context

get_context("fork")
loadFile = "./bubble_demo.yaml"
mypath = os.path.abspath(os.curdir)
print("Absolute path : {}".format(mypath))


nexus = Nexus("Nexus")
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
