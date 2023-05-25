import pynwb
from nwbinspector.tools import get_s3_urls_and_dandi_paths
from improv.actor import Actor, RunManager
from datetime import date  # used for saving
import numpy as np
import traceback
import glob

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

 
class Generator(Actor):
    """ Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator"
        self.frame_num = 0
        self.io = None
        self.nwbfile = None
        self.max_frame = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        logger.info('Beginning setup for Generator')
        path = "demos/nwb_local/sub-222549_ecephys+ophys.nwb"
        try:
            logger.info('Opening local file')
            self.io = pynwb.NWBHDF5IO(path, mode='r')
            self.nwbfile = self.io.read()
            logger.info('Completed opening file')
            self.max_frame = self.nwbfile.acquisition['TwoPhotonSeries_green'].data.shape[0]
        except Exception as e:
            logger.error(f"-------------------Generator Exception: {e}")
            logger.error(traceback.format_exc())
        
        logger.info('Completed setup for Generator')

    def stop(self):
        """ Save current randint vector to a file.
        """

        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        if (self.frame_num >= self.max_frame):
            logger.warning("Warning: All available data have been read")
        else:
            try:
                self.data = self.nwbfile.acquisition['TwoPhotonSeries_green'].data[self.frame_num, :,:]
                data_id = self.client.put(
                    self.data, str(f"Gen_raw: {self.frame_num}"))
                self.q_out.put([[data_id, str(self.frame_num)]])
                logger.info(f"Generated frame at {self.frame_num}")
                self.frame_num += 1
            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")
                logger.error(traceback.format_exc())