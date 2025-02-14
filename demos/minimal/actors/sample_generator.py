from improv.actor import Actor
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate a sine/cosine wave based on frame number to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Generator"
        self.data = None
        self.frame_num = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Generates an array that serves as an initial source of data.

        Initial data is a 2D cosine wave consisting of 100 evenly spaced xy points ranging from -10 to 10 inclusive.
        """
        logger.info("Beginning setup for Generator")

        # generate 100 evenly spaced values from -10 to 10
        xs = np.linspace(-10, 10, 100)
        ys = np.cos(xs)
        # stack xs and ys to create a (100, 2) array of xy points
        self.data = np.column_stack([xs, ys])

        logger.info("Completed setup for Generator")

    def stop(self):
        """Save current wave vector to file."""
        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.

        If the frame number is odd, the data is a sine wave. If the frame number is even, the data is a cosine wave.
        """
        # set a max number of frames to generate
        if self.frame_num > 1000:
            return

        xs = np.linspace(-10, 10, 100)

        # Generate sine or cosine values based on frame number
        if self.frame_num % 2 == 1:
            ys = np.sin(xs)
        else:
            ys = np.cos(xs)

        # update data
        self.data = np.column_stack([xs, ys])

        # create flattened array with x and y coordinates along with the current frame number
        data = np.append(self.data.ravel(), self.frame_num)  # Shape (201,)

        # Send the flattened array with frame_num
        try:
            data_id = self.client.put(data)
            if self.store_loc:
                self.q_out.put([[data_id, str(self.frame_num)]])
            else:
                self.q_out.put(data_id)

            # Increment frame number
            self.frame_num += 1
        except Exception as e:
            logger.error(f"--------------------------------Generator Exception: {e}")



