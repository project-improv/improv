from improv.actor import Actor, RunManager
import numpy as np
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Tester(Actor):
    """ Actor intended for use within a testing environment.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isSetup = False
        self.hasRun = False
        self.hasStopped = False
        self.runNum = 0
        
    def setup(self):
        self.isSetup = True
    
    def run(self):
        self.hasRun = True
        with RunManager('tester', self.runMethod, self.setup, self.q_sig, self.q_comm, self.stop) as rm:
            logger.info(rm)
    
    def runMethod(self):
        self.runNum += 1
    
    def stop(self):
        self.hasStopped = True
    
    