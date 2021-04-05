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

# else:
       
    
       
       # newparent_directory = os.path.split(parent_directory)[0]# Repeat as needed  
       #file_path = os.path.join(newparent_directory, 'demodata') 
       # print("datafile '%s'" %file_path)
       # if myfile not in os.listdir(file_path):
           # raise FileNotFoundError
       # else: 
            #file_path = os.path.join(file_path, myfile)
 path = os.path.join(os.getcwd(), "data" ) 
 print("path '%s'" %path)
 if not os.path.exists(path):
    os.mkdir(path)
    print("Directory '%s' created" %path)
    print("Place file into data directory for analysis")  




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
