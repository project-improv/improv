from improv.actor import Actor
from queue import Empty
import logging
import zmq
import numpy as np
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """Sample processor used to scale a sine or cosine wave and calculate the amplitude.
    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initializes all class variables.

        self.name (string): name of the actor.
        self.frame (ObjectID): StoreInterface object id referencing data from the store.
        self.frame_num (int): index of current frame.
        """
        self.name = "Processor"
        self.frame = None
        self.frame_num = 0

        logger.info("Completed setup for Processor")

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info("Processor stopping")
        return 0

    def runStep(self):
        """
        Gets from the input queue, scales the data in the y-dimension by a random number between 1-10 inclusive and then
        calculates the amplitude of the wave.

        """
        data_id = None
        try:
            data_id = self.q_in.get(timeout=0.05)
        except Exception:
            logger.error(f"Could not get frame!")
            pass

        if data_id is not None:
            try:
                if self.store_loc:
                    # Fetch the data from the client using the ObjectID
                    self.frame = self.client.getID(data_id[0][0])
                else:
                    self.frame = self.client.get(data_id)

                # Unpack the frame to get the data and frame number
                data = np.array(self.frame, dtype=np.float64)
                self.frame_num = int(data[-1])
                # reshape the data to 2D array
                data = data[:-1].reshape(-1, 2)

                # Scale the y-values of the sine or cosine wave by random factor
                scale_factor = random.randint(1, 10)
                data[:, 1] *= scale_factor

                # calculate the amplitude and frequency
                amplitude = np.round((data.max(axis=0)[1] - data.min(axis=0)[1]) / 2)
                logger.info(f"Frame {self.frame_num} has amplitude {amplitude}")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")

