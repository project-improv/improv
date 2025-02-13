from improv.actor import Actor
import random
import logging
import zmq
import numpy as np
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """Sample processor used to scale a sine or cosine wave and calculate the amplitude.
    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initializes all class variables. Create and bind the socket for zmq to send to fastplotlib.ipynb
        for visualization.

        self.name (string): name of the actor.
        self.frame (ObjectID): StoreInterface object id referencing data from the store.
        self.frame_num (int): index of current frame.
        self.processed_data (np.array): raveled array containing the processed data appended with the current frame number
        """
        self.name = "Processor"
        self.frame = None
        self.frame_num = 0
        self.processed_data = None

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        logger.info("Completed setup for Processor")

    def stop(self):
        """Stop function that closes the zmq socket to visualization notebook."""
        logger.info("Processor stopping")
        self.socket.close()
        return 0

    def runStep(self):
        """
        Gets from the input queue, scales the data in the y-dimension by a random number between 1-10 inclusive and then
        calculates the amplitude.
        """
        data_id = None
        try:
            data_id = self.q_in.get(timeout=0.05)
        except Exception as e:
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

                # calculate the amplitude
                amplitude = np.round((data.max(axis=0)[1] - data.min(axis=0)[1]) / 2)
                logger.info(f"Frame {self.frame_num} has amplitude {amplitude}")

                # Flatten processed values and append frame number
                self.processed_data = np.append(data.ravel(), self.frame_num)

                # slight pause for visualization
                time.sleep(2)

                logger.info("Sending data to visualization notebook!")
                # Send the processed data through the ZMQ socket
                self.socket.send(self.processed_data.tobytes())

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
