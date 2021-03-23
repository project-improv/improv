import logging
import os
import shutil
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from improv.nexus import Nexus


loadFile = './basic_demo.yaml'

nexus = Nexus('Nexus')
nexus.createNexus(file=loadFile)

#creates output directory for data
directory = "improv_output" 
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory) 
if not os.path.exists(path):
    os.mkdir(path)
print("Directory '%s' created" %path)


current_directory = os.path.dirname(basic_demo.py)

print("Directory '%s' is current directory" %current_directory)

parent_directory = os.path.split(current_directory)[0] # Repeat as needed
newparent_directory = os.path.split(parent_directory)[0] # Repeat as needed
print("Directory '%s' is parent directory" %newparent_directory)
file_path = os.path.join(newparent_directory, 'demodata/Tolias_mesoscope_2.hdf5')
file1 = open(file_path)


directory = "data" 
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory) 
if not os.path.exists(path):
    os.mkdir(path)
# if the file is not already in dir
if "Tolias_mesoscope_2.hdf5" not in path.namelist():
    shutil.move(path, file1)
print("File '%s' created" %file1)


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
