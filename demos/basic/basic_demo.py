import logging
import os
import shutil
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from improv.nexus import Nexus


loadFile = './basic_demo.yaml'

nexus = Nexus('Nexus')
nexus.createNexus(file=loadFile)

# #creates output directory for data
# directory = "output" 
# parent_dir = os.getcwd()
# path = os.path.join(parent_dir, directory) 
# if not os.path.exists(path):
#     os.mkdir(path)
#     print("Directory '%s' created" %path)
# path = os.path.join(path, "timing")
# if not os.path.exists(path):
#     os.mkdir(path)
#     print("Directory '%s' created" %path)




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
