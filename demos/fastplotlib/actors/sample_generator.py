from improv.actor import Actor
import numpy as np
import logging;

logger = logging.getLogger(__name__)
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
        xs = np.linspace(-10, 10, 100)
        if self.frame_index % 2 == 0:
            ys = np.cos(xs)
            data = np.dstack((xs, ys))[0]
        else:
            ys = np.sin(xs)
            data = np.dstack((xs, ys))[0]

        frame_ix = np.array([self.frame_index], dtype=np.float64)

        # there must be a better way to do this
        out = np.concatenate(
            [data.ravel(), frame_ix],
            dtype=np.float64
        )

        self.q_out.put(out)

        self.frame_index += 1
