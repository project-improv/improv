from improv.actor import Actor, RunManager
import numpy as np
import logging
import time  # Importing time module for the delay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor generate sine/cosine waves based on odd/even frame numbers respectively to pass into a sample processor.
    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Generator"
        self.frame_num = 0  # Initialize frame counter
        self.data = None
        self.max_frames = 20  # Set the limit for number of frames

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Initial setup for Generator"""
        logger.info("Beginning setup for Generator")
        xs = np.linspace(-10, 10, 100)
        ys = np.sin(xs)
        self.data = np.column_stack((xs, ys))
        logger.info("Completed setup for Generator")

    def stop(self):
        """Save current wave vector to file."""
        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.
        
        Data is a sine wave if the frame number is even or a cosine wave if the frame number is odd."""
        time.sleep(0.5) # Add a slight pause between frame generation
        """Sends a flattened array with x and y coordinates followed by frame number."""
        if self.frame_num >= self.max_frames:
            logger.info(f"Reached maximum frame count ({self.max_frames}). Stopping generation.")
            return

        xs = np.linspace(-10, 10, 100)

        # Generate sine or cosine values based on frame number
        if self.frame_num % 2 == 0:
            # Even frame: Generate sine wave
            ys = np.sin(xs)
            # wave_type = "sine"
        else:
            # Odd frame: Generate cosine wave
            ys = np.cos(xs)
            # wave_type = "cosine"

        # Combine x and y into a 2D array
        data = np.column_stack((xs, ys))  # Shape (100, 2)

        # Flatten the 2D array to 1D
        flattened_values = data.flatten()  # Shape (200,)

        # Append frame_num as the last element
        data_to_send = np.append(flattened_values, self.frame_num)  # Shape (201,)

        # Send the flattened array with frame_num
        try:
            data_id = self.client.put(data_to_send, f"Frame: {self.frame_num}")
            self.q_out.put([[data_id, f"Frame: {self.frame_num}"]])
            # logger.info(f"Sent frame {self.frame_num} with flattened x and y coordinates ({wave_type} wave)")
        except Exception as e:
            logger.error(f"Generator Exception: {e}")

        # Increment frame number
        self.frame_num += 1
