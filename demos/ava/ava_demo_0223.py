import logging
# Matplotlib is overly verbose by default
logging.getLogger("matplotlib").setLevel(logging.INFO)

if __name__ == '__main__':
    '''
    '''
    from improv.nexus import Nexus

    loadFile = 'ava_demo_0223.yaml'

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