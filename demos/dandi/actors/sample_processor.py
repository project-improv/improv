from improv.actor import Actor
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """ Sample processor used to calculate the average of an array of integers.

    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """ Initializes all class variables.

        self.name (string): name of the actor.
        self.frame (ObjectID): Store object id referencing data from the store.
        self.avg_list (list): list that contains averages of individual vectors.
        self.frame_num (int): index of current frame.
        """

        self.name = "Processor"
        self.frame = None
        self.avg_list = []
        self.frame_num = 1

        self.behave = None
        self.behave_list = []
        self.behave_num = 1
        logger.info('Completed setup for Processor')

    def stop(self):
        """ Trivial stop function for testing purposes.
        """

        logger.info("Processor stopping")

    def runStep(self):
        """ Gets from the input queue and calculates the average.

        Receives an ObjectID, references data in the store using that
        ObjectID, calculates the average of that data, and finally prints 
        to stdout. 
        """
        frame = None
        try:
            frame = self.q_in.get(timeout=0.001)

        except:
            logger.error("Could not get frame!")
            time.sleep(1)
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame = self.client.getID(frame[0][0])
            avg = np.mean(self.frame[0])

            logger.info(f"Average: {avg}")
            self.avg_list.append(avg)
            logger.info(f"Overall Average: {np.mean(self.avg_list)}")
            logger.info(f"Frame number: {self.frame_num}")
            self.frame_num += 1
        
        behave = None
        try:
            behave = self.links['bq_in'].get(timeout=0.001)

        except:
            logger.error("Could not get position!")
            time.sleep(1)
            pass

        if behave is not None and self.behave_num is not None:
            self.done = False
            self.behave = self.client.getID(behave[0][0])
            self.behave_list.append(self.behave)
            logger.info(f"Standard Deviation of Position: {np.std(self.behave_list)}")
            logger.info(f"Behavior number: {self.behave_num}")
            self.behave_num += 1