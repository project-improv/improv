from improv.actor import Actor
from queue import Empty
import logging
import zmq
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """
    Process data by scaling y coordinates by 2 and send it through zmq to be visualized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """
        Creates and binds the socket for zmq and initializes processed data storage.
        """
        self.name = "Processor"
        self.processed_data = None  # Initialize variable to store processed data
        self.frame = None  # Initialize variable to store the current frame

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        logger.info("Completed setup for Processor")

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info("Processor stopping")
        self.socket.close()
        return 0

    def runStep(self):
        """
        Receives data ID from the queue, retrieves data from the Plasma store,
        processes it, stores it in `processed_data`, and sends it through the
        socket as flattened data with the frame number appended.
        """
        try:
            # Retrieve data ID from the queue
            data_id = self.q_in.get(timeout=0.05)
        except Empty:
            return  # No data received, skip this step
        except Exception as e:
            logger.error(f"Error retrieving data ID: {e}")
            return

        if data_id is not None:
            try:
                # Fetch the data from the client using the ObjectID
                self.frame = self.client.getID(data_id[0][0])  # Retrieve the frame data

                # Unpack the flattened data
                data = np.array(self.frame, dtype=np.float64)  # Ensure it's a NumPy array
                frame_num = int(data[-1])  # Extract the last element as frame number
                data = data[:-1].reshape(-1, 2)  # Exclude the last element

                # Perform processing (e.g., scaling the y-values)
                data[:, 1] *= 2  # Example: Scale y-coordinates by 2

                # Flatten processed values and append frame number
                self.processed_data = np.append(np.ravel(data), frame_num)


                # Send the processed data through the ZMQ socket
                self.socket.send(self.processed_data.tobytes())
                logger.info(f"Frame {frame_num}: Sent {data.shape[0]} points after processing")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
