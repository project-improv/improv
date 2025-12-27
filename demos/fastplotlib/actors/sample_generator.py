from improv.actor import Actor, RunManager
from datetime import date #used for saving
import numpy as np
import time
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """
    Generate data and puts it in the queue for the processor to take
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator"
        self.frame_index = 0
    
    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        logger.info('Completed setup for Generator')

    def stop(self):
        print("Generator stopping")
        return 0

    def runStep(self):
        """
        Generates 512 x 512 frame and puts it in the queue for the processor
        """

        data = np.random.randint(0, 255, size=(512 * 512), dtype=np.uint16).reshape(512, 512)

        frame_ix = np.array([self.frame_index], dtype=np.uint32)

        # there must be a better way to do this
        out = np.concatenate(
            [data.ravel(), frame_ix],
            dtype=np.uint32
        )

        self.q_out.put(out)

        self.frame_index += 1
