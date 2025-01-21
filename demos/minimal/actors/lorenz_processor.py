from improv.actor import Actor
from queue import Empty
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
        Sets up the ZMQ socket and initializes storage for processed data.
        """
        self.name = "Processor"
        self.processed_data = None  # Storage for processed data
        self.frame = None  # Storage for the current frame

        # Set up ZMQ PUB socket
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://127.0.0.1:5555")

        logger.info("Processor setup completed. ZMQ PUB socket bound to tcp://127.0.0.1:5555")

    def stop(self):
        """
        Trivial stop function for testing purposes.
        """
        logger.info("Processor stopping")
        self.socket.close()
        return 0

    def runStep(self):
        """
        Processes incoming Lorenz data, applies transformations, and sends
        the processed data through a ZMQ socket.
        """
        try:
            # Retrieve data ID from the input queue
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

                # Convert the frame data to a NumPy array
                data = np.array(self.frame, dtype=np.float64)  # Ensure it's a NumPy array
                frame_num = int(data[-1])  # Extract the last element as the frame number
                data = data[:-1].reshape(-1, 3)  # Exclude the last element (frame number)
                

                # Perform processing on the Lorenz coordinates
                # Example 1: Scale x-coordinates by 0.5 and y-coordinates by 2
                data[:, 0] *= 2  # Scale x-coordinates
                data[:, 1] *= 2  # Scale y-coordinates
                data[:, 2] *= 2  # Scale z-coordinates

                # Example 2: Add sinusoidal noise to the y-coordinates
                data[:, 1] += np.sin(data[:, 0])

                # Flatten processed values and append frame number
                self.processed_data = np.append(np.ravel(data), frame_num)

                # Send the processed data through the ZMQ socket
                self.socket.send(self.processed_data.tobytes())
                logger.info(f"Frame {frame_num}: Sent points with size {data.shape} after processing")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
