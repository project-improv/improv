import random
import time

from improv.actor import ZmqActor
from datetime import date  # used for saving
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(ZmqActor):
    """Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, output_filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator"
        self.frame_num = 0
        self.output_filename = output_filename

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Generates an array that serves as an initial source of data.

        Initial array is a 100 row, 5 column numpy matrix that contains
        integers from 1-99, inclusive.
        """

        self.data = np.asmatrix(np.random.randint(100, size=(100, 5)))
        self.improv_logger.info("Completed setup for Generator")

    def stop(self):
        """Save current randint vector to a file."""

        self.improv_logger.info("Generator stopping")
        return 0

    def run_step(self):
        """Generates additional data after initial setup data is exhausted.

        Data is of a different form as the setup data in that although it is
        the same size (5x1 vector), it is uniformly distributed in [1, 10]
        instead of in [1, 100]. Therefore, the average over time should
        converge to 5.5.
        """


        device_time = time.time_ns()
        time.sleep(0.0003)
        acquired_time = time.time_ns()  # mock the time the generator "actually received" the data

        if self.frame_num < np.shape(self.data)[0]:

            with open(self.output_filename, "a+") as f:
                device_data = self.data[self.frame_num]
                packaged_data = (acquired_time, (device_time, device_data))
                # save the data to the flat file before sending it downstream
                f.write(f"{packaged_data[0]}, {packaged_data[1][0]}, {packaged_data[1][1]}\n")

                data_id = self.client.put(packaged_data)
                try:
                    self.q_out.put(data_id)
                    # self.improv_logger.info(f"Sent {self.data[self.frame_num]} with key {data_id}")
                    self.frame_num += 1

                except Exception as e:
                    self.improv_logger.error(f"Generator Exception: {e}")
        else:
            self.data = np.concatenate(
                (self.data, np.asmatrix(np.random.randint(10, size=(1, 5)))), axis=0
            )
