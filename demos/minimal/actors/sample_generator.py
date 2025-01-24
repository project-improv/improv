from improv.actor import Actor, RunManager
import numpy as np
import logging
import time  # Importing time module for the delay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate a sine/cosine wave based on frame number to pass into a sample processor. Odd frames generate a sine wave and even frames generate a cosine wave.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Generator"
        self.data = None
        self.max_frames = 500  # Set the limit for number of frames

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Initializes all class variables.

            self.data (ndarray): 2D NumPy array where the first column is x-values 
                                (linearly spaced between -10 and 10) and the second column 
                                is the sine of these x-values.
            self.frame_num (int): index of the current frame, initialized to 0.
        """
        logger.info("Beginning setup for Generator")
        xs = np.linspace(-10, 10, 100)
        ys = np.sin(xs)
        self.data = np.column_stack((xs, ys))
        self.frame_num = 0  # Initialize frame counter
        logger.info("Completed setup for Generator")


    def stop(self):
        """Save current wave vector to file."""
        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.
        
        Data is a sine wave if the frame number is odd or a cosine wave if the frame number is even."""
        time.sleep(0.5) # Add a slight pause between frame generation

        #Sends a flattened array with x and y coordinates followed by frame number.
        if self.frame_num >= self.max_frames:
            logger.info(f"Reached maximum frame count ({self.max_frames}). Stopping generation.")
            return

        xs = np.linspace(-10, 10, 100)

        # Generate sine or cosine values based on frame number
        if self.frame_num % 2 == 1:
            # Even frame: Generate sine wave
            ys = np.sin(xs)
        else:
            # Odd frame: Generate cosine wave
            ys = np.cos(xs)

        # Combine x and y into a 1D array, append frame_num
        data_to_send = np.append(np.ravel(np.column_stack((xs, ys))), self.frame_num)  # Shape (201,)

        # Send the flattened array with frame_num
        try:
            data_id = self.client.put(data_to_send, f"Frame: {self.frame_num}")
            self.q_out.put([[data_id, f"Frame: {self.frame_num}"]])
        except Exception as e:
            logger.error(f"Generator Exception: {e}")

        # Increment frame number
        self.frame_num += 1
