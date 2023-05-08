from improv.actor import Actor, RunManager
from datetime import date  # used for saving
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate data to pass into a sample processor.

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

        Initial array is a 100 row, 5 column numpy matrix that contains
        integers from 1-99, inclusive.
        """

        self.data = np.asmatrix(np.random.randint(100, size=(100, 5)))
        logger.info('Completed setup for Generator')

    # def run(self):
    #     """ Send array into the store.
    #     """
    #     self.fcns = {}
    #     self.fcns['setup'] = self.setup
    #     self.fcns['run'] = self.runStep
    #     self.fcns['stop'] = self.stop

    #     with RunManager(self.name, self.fcns, self.links) as rm:
    #         logger.info(rm)

    def stop(self):
        """Save current randint vector to a file."""

        print("Generator stopping")
        np.save(
            f"sample_generator_data_{date.today()}", self.data
        )  # This is not the best example of a save function, will overwrite previous files with the same name.
        return 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.

        Data is of a different form as the setup data in that although it is
        the same size (5x1 vector), it is uniformly distributed in [1, 10]
        instead of in [1, 100]. Therefore, the average over time should
        converge to 5.5.
        """

        if self.frame_num < np.shape(self.data)[0]:
            data_id = self.client.put(
                self.data[self.frame_num], str(f"Gen_raw: {self.frame_num}")
            )
            try:
                self.q_out.put([[data_id, str(self.frame_num)]])
                self.frame_num += 1
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
        else:
            self.data = np.concatenate(
                (self.data, np.asmatrix(np.random.randint(10, size=(1, 5)))), axis=0
            )
