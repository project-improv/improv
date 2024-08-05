from improv.actor import ZmqActor
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(ZmqActor):
    """Sample processor used to calculate the average of an array of integers.

    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]

    def setup(self):
        """Initializes all class variables.

        self.name (string): name of the actor.
        self.frame (ObjectID): StoreInterface object id referencing data from the store.
        self.avg_list (list): list that contains averages of individual vectors.
        self.frame_num (int): index of current frame.
        """

        if not hasattr(self, "name"):
            self.name = "Processor"
        self.frame = None
        self.avg_list = []
        self.frame_num = 1
        self.improv_logger.info("Completed setup for Processor")

    def stop(self):
        """Trivial stop function for testing purposes."""

        self.improv_logger.info("Processor stopping")
        return 0

    def run_step(self):
        """Gets from the input queue and calculates the average.

        Receives an ObjectID, references data in the store using that
        ObjectID, calculates the average of that data, and finally prints
        to stdout.
        """

        frame = None
        try:
            frame = self.q_in.get(timeout=0.05)
        except Exception as e:
            # logger.error(f"{self.name} could not get frame! At {self.frame_num}: {e}")
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame = self.client.get(frame)
            avg = np.mean(self.frame[0])
            # self.improv_logger.info(f"Average: {avg}")
            self.avg_list.append(avg)
            # self.improv_logger.info(f"Overall Average: {np.mean(self.avg_list)}")
            # self.improv_logger.info(f"Frame number: {self.frame_num}")
            self.frame_num += 1
