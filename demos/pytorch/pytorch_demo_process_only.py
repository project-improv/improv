import logging
import time
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
from improv.nexus import Nexus

loadFile = './pytorch_demo_process_only.yaml'

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

nexus.setup()
time.sleep(100)
nexus.run()
time.sleep(100)
nexus.quit()