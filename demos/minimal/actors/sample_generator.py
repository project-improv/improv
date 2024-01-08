from improv.actor import Actor, RunManager
from datetime import date  # used for saving
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py. For visualization using `fastplotlib`,
    see /demos/sample_actors/visual/.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator"
        self.frame_num = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        logger.info("Completed setup for Generator")

    def stop(self):
        """Save current cosine or sine wave vector to a file."""

        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """Generates a sine or cosine wave based on the frame_num and puts it into a queue
        for the processor to take.
        """
        xs = np.linspace(-10, 10, 100)
        if self.frame_num % 2 == 0:
            ys = np.cos(xs)
            self.data = np.dstack((xs, ys))[0]
        else:
            ys = np.sin(xs)
            self.data = np.dstack((xs, ys))[0]

        # put current data in the data store
        data_id = self.client.put(
                self.data.ravel(), str(f"Gen_raw: {self.frame_num}")
            )
            # logger.info('Put data in store')
        try:
            self.q_out.put([[data_id, str(self.frame_num)]])
            logger.info("Sent message on")
            self.frame_num += 1
        except Exception as e:
            logger.error(
                f"--------------------------------Generator Exception: {e}"
            )

