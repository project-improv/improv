from improv.actor import Actor, RunManager
from datetime import date  # used for saving
import numpy as np
import logging
import time  # Importing time module for the delay

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """
    Generates coordinates for the Lorenz system in real time.
    Computes the next Lorenz coordinates every half second.
    Outputs a flattened array with progressively filled 100 (x, y) pairs
    and appends the frame number at the end.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None  # Placeholder for the 2D array
        self.coordinates = []
        self.name = "lorenz_generator"
        self.dt = 0.01  # Time step for numerical integration
        self.max_points = 1000  # Total number of (x, y) pairs
        self.points_per_frame = 10  # Number of points to add per frame

    def __str__(self):
        return f"Name: {self.name}, Current Coordinates: {self.coordinates[-1] if self.coordinates else None}"

    def setup(self):
        """
        Initializes the Lorenz system with the starting coordinates and the data array.
        """
        logger.info("Beginning setup for LorenzGenerator")
        initial_coordinate = np.array([1.0, 1.0, 1.0])  # Initial coordinates (x, y, z)
        self.coordinates = [initial_coordinate]

        # Initialize the data array as a 2D array (100 rows for x, y coordinates)
        self.data = np.zeros((self.max_points, 2))  # Shape (100, 2)
        self.frame_num = 0
        self.current_index = 0  # Tracks the current position to fill in the data arrayx
        logger.info(f"Initialized Lorenz system with initial coordinates: {initial_coordinate}")

    def stop(self):
        """
        Save the last Lorenz coordinates to a file for persistence.
        """
        logger.info("LorenzGenerator stopping")
        np.save("lorenz_last_coordinate.npy", self.coordinates[-1])
        return 0

    def runStep(self):
        """
        Generates the next 10 Lorenz coordinates and fills them into the 2D data array.
        Sends the progressively filled data as a flattened array with the frame number appended.
        """
        time.sleep(0.5)  # Delay for half a second

        try:
            # Add the next 10 points
            for _ in range(self.points_per_frame):
                if self.current_index >= self.max_points:
                    logger.info(f"Data array fully filled for frame {self.frame_num}.")
                    break  # Stop filling if all 100 points are generated

                # Compute the next coordinate
                derivative = lorenz(self.coordinates[-1])
                next_coordinate = self.coordinates[-1] + derivative * self.dt
                self.coordinates.append(next_coordinate)

                # Fill x and y coordinates into the 2D data array
                self.data[self.current_index, 0] = next_coordinate[0]  # x-coordinate
                self.data[self.current_index, 1] = next_coordinate[1]  # y-coordinate

                self.current_index += 1  # Increment the index for the next point

            # Flatten the 2D array and append the frame number
            flattened_data = self.data.flatten()  # Shape (200,)
            data_to_send = np.append(flattened_data, self.frame_num)  # Shape (201,)

            # Send the data
            data_id = self.client.put(data_to_send, str(f"Lorenz_Frame: {self.frame_num}"))
            self.q_out.put([[data_id, str(self.frame_num)]])
            logger.info(f"Generated Lorenz frame {self.frame_num} with progressively filled data array.")

            # Increment frame number for the next step
            self.frame_num += 1
        except Exception as e:
            logger.error(f"LorenzGenerator Exception: {e}")


def lorenz(xyz, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])
