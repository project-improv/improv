from improv.actor import Actor, RunManager
import numpy as np
from queue import Empty
import logging; logger = logging.getLogger(__name__)
import zmq
logger.setLevel(logging.INFO)


class Processor(Actor):
    """
    Process data and send it through zmq to be be visualized
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self):
        """
        Creates and binds the socket for zmq
        """

        self.name = "Processor"

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        self.frame_index = 0

        logger.info('Completed setup for Processor')

    def stop(self):
        logger.info("Processor stopping")
        return 0

    def runStep(self):
        """
        Gets the frame from the queue, take the mean, sends a memoryview
        so the zmq subscriber can get the buffer to update the plot
        """

        frame = None

        try:
            frame = self.q_in.get(timeout=0.05)
        except Empty:
            pass
        except:
            logger.error("Could not get frame!")

        if frame is not None:
            self.frame_index += 1
            # do some processing
            frame.mean()
            # send the buffer
            self.socket.send(frame)
