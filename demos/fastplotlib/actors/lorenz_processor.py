from improv.actor import Actor
import time
import logging
import zmq
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """
    Processes Lorenz data by performing custom transformations on the coordinates
    (e.g., scaling and applying mathematical operations) and sends the processed
    data through a ZMQ socket for visualization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """
        Sets up the ZMQ socket and initialize class variables.
        """
        self.name = "Processor"
        self.frame = None
        self.frame_num = None

        # Set up ZMQ PUB socket
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        logger.info("Processor setup completed. ZMQ PUB socket bound to tcp://127.0.0.1:5555")

    def stop(self):
        """
        Stop function. Closes the ZMQ socket connection.
        """
        logger.info("Processor stopping")
        self.socket.close()
        return 0

    def runStep(self):
        """
        Trivial processing step that gets the lorenz data and passes it through the
        ZMQ socket for visualization.
        """
        # Delay for half a second for visualization purposes
        time.sleep(0.5)

        data_id = None
        try:
            data_id = self.q_in.get(timeout=0.05)
        except Exception:
            logger.error(f"Could not get frame!")
            pass

        if data_id is not None:
            try:
                if self.store_loc:
                    # Fetch the data from the client using the ObjectID
                    self.frame = self.client.getID(data_id[0][0])
                else:
                    self.frame = self.client.get(data_id)


                # unpack the frame to get the frame number
                data = np.array(self.frame, dtype=np.float64)
                self.frame_num = int(data[-1])


                # Send the processed data through the ZMQ socket
                self.socket.send(data)
                logger.info(f"Frame {self.frame_num}: Sent points with size {data.shape} after processing")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")