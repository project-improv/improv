from improv.nexus import Nexus

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

loadFile = "./configs/sample_config.yaml"

nexus = Nexus("Sample")
nexus.createNexus(file=loadFile)
nexus.startNexus()
