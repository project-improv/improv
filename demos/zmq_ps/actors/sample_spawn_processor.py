from improv.actor import Actor, RunManager
import numpy as np
from queue import Empty
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """Sample processor used to calculate the average of an array of integers.

    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initializes all class variables.

        self.name (string): name of the actor.
        self.frame (ObjectID): Store object id referencing data from the store.
        self.avg_list (list): list that contains averages of individual vectors.
        self.frame_num (int): index of current frame.
        """

        self.name = "Processor"
        self.frame = None
        self.avg_list = []
        self.frame_num = 1
        logger.info("Completed setup for Processor")

        self._getStoreInterface()

    def stop(self):
        """Trivial stop function for testing purposes."""

        logger.info("Processor stopping")
        return 0

    # def run(self):
    #     """ Send array into the store.
    #     """
    #     self.fcns = {}
    #     self.fcns['setup'] = self.setup
    #     self.fcns['run'] = self.runStep
    #     self.fcns['stop'] = self.stop

    #     with RunManager(self.name, self.fcns, self.links) as rm:
    #         logger.info(rm)

    def runStep(self):
        """Gets from the input queue and calculates the average.

        Receives an ObjectID, references data in the store using that
        ObjectID, calculates the average of that data, and finally prints
        to stdout.
        """

        frame = None
        try:
            frame = self.q_in.get(timeout=0.05)
        except Empty:
            pass
        except:
            logger.error("Could not get frame!")
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame = self.client.getID(frame[0][0])
            avg = np.mean(self.frame[0])
            print(f"Average: {avg}")
            self.avg_list.append(avg)
            print(f"Overall Average: {np.mean(self.avg_list)}")
            print(f"Frame number: {self.frame_num}")
            self.frame_num += 1
