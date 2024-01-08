import numpy as np
import logging

from demos.sample_actors.zmqActor import ZmqActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Generator(ZmqActor):
    """Sample actor to generate data to pass into a sample processor
    using async ZMQ to communicate.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator"
        self.frame_num = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Generates an array that serves as an initial source of data.
        Sets up a ZmqRRActor to send data to the processor.

        Initial array is a 100 row, 5 column numpy matrix that contains
        integers from 1-99, inclusive.
        """
        logger.info("Beginning setup for Generator")
        self.data = np.asmatrix(np.random.randint(100, size=(100, 5)))
        logger.info("Completed setup for Generator")

    def stop(self):
        """Save current randint vector to a file."""
        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.
        Sends data to the processor using a ZmqRRActor.

        Data is of a different form as the setup data in that although it is
        the same size (5x1 vector), it is uniformly distributed in [1, 10]
        instead of in [1, 100]. Therefore, the average over time should
        converge to 5.5.
        """
        if self.frame_num < np.shape(self.data)[0]:
            data_id = self.client.put(self.data[self.frame_num], str(f"Gen_raw_{self.frame_num}"))
            try:
                self.put(data_id)
                self.frame_num += 1
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
        else:
            self.data = np.concatenate((self.data, np.asmatrix(np.random.randint(10, size=(1, 5)))), axis=0)
