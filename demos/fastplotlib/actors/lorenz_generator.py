from improv.actor import Actor
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Generator(Actor):
    """
    Generates coordinates for the Lorenz system in real time.
    Computes the next 50 Lorenz coordinates every half second.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Lorenz Generator"
        self.dt = 0.01  # time step for numerical integration
        self.frame_num = 0

    def __str__(self):
        return f"Name: {self.name}, Current Coordinates: {self.coordinates[-1] if self.coordinates else None}"

    def setup(self):
        """Generates an array that serves as an initial source of data.

        Initial data is the starting point (x,y,z) for the lorenz attractor.
        """
        logger.info("Beginning setup for Lorenz Generator")

        # initial condition
        self.data = np.array([1.0, 1.0, 1.0]).reshape(1,3)

        logger.info(f"Initialized Lorenz system with initial coordinates: {self.data}")

    def stop(self):
        """Trivial stop function."""
        logger.info("Lorenz Generator stopping")
        return 0

    def runStep(self):
        """Generates the next 25 points in the lorenz system.

        Sends the progressively filled data as a flattened array with the frame number appended to the processor.
        """
        if self.frame_num >= 150:
            return

        # Add the next 10 points
        for _ in range(25):
            # Compute the next coordinate
            derivative = lorenz(self.data[-1])
            next_coordinate = self.data[-1] + derivative * self.dt
            self.data = np.vstack((self.data, next_coordinate))

        # create flattened array with xyz coordinates along with the current frame number
        data = np.append(self.data.ravel(), self.frame_num)

        # Send the flattened array with frame_num
        try:
            data_id = self.client.put(data)
            if self.store_loc:
                self.q_out.put([[data_id, str(self.frame_num)]])
            else:
                self.q_out.put(data_id)

            # Increment frame number
            self.frame_num += 1
        except Exception as e:
            logger.error(f"--------------------------------Generator Exception: {e}")


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
