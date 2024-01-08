import numpy as np
import logging

from demos.sample_actors.zmqActor import ZmqActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(ZmqActor):
    """Sample processor used to calculate the average of an array of integers
    using sync ZMQ to communicate.

    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initializes all class variables.
        Sets up a ZmqRRActor to receive data from the generator.

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

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info("Processor stopping; have received {} frames so far".format(self.frame_num))

    def runStep(self):
        """Gets from the input queue and calculates the average.
        Receives data from the generator using a ZmqRRActor.

        Receives an ObjectID, references data in the store using that
        ObjectID, calculates the average of that data, and finally prints
        to stdout.
        """

        frame = None
        try:
            frame = self.get()

        except:
            logger.error("Could not get frame!")
            pass

        if frame is not None:
            self.done = False
            self.frame = self.client.getID(frame)
            avg = np.mean(self.frame[0])

            # logger.info(f"Average: {avg}")
            self.avg_list.append(avg)
            logger.info(f"Overall Average: {np.mean(self.avg_list)}")
            # logger.info(f"Frame number: {self.frame_num}")
            self.frame_num += 1
