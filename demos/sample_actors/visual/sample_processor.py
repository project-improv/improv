from improv.actor import Actor
from queue import Empty
import logging; logger = logging.getLogger(__name__)
import zmq
logger.setLevel(logging.INFO)
import numpy as np
import torch


class Processor(Actor):
    """
    Process data and send it through zmq to be visualized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """
        Creates and binds the socket for zmq.
        """

        self.name = "Processor"

        gpu_available = torch.cuda.is_available()

        if not gpu_available:
            logger.error("GPU is needed for fastplotlib visualization.")

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        self.frame_index = 1

        logger.info('Completed setup for Processor')

    def stop(self):
        logger.info("Processor stopping")
        self.socket.close()
        return 0

    def runStep(self):
        """
        Gets the data_id to the store from the queue, fetches the frame from the data store,
        take the mean, sends a memoryview so the zmq subscriber can get the buffer to update
        the plot.
        """

        frame = None

        try:
            frame = self.q_in.get(timeout=0.05)
        except Empty:
            pass
        except:
            logger.error("Could not get frame!")

        if frame is not None:
            # get frame from data store
            self.frame = self.client.getID(frame[0][0])

            # do some processing
            self.frame.mean()

            frame_ix = np.array([self.frame_index], dtype=np.float64)

            # send the buffer data and frame number as an array
            out = np.concatenate(
            [self.frame, frame_ix],
            dtype=np.float64
            )
            self.frame_index += 1
            self.socket.send(out)
