from fsspec.registry import known_implementations
import fsspec
import pynwb
import h5py
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
from nwbinspector.tools import get_s3_urls_and_dandi_paths
from improv.actor import Actor, RunManager
from datetime import date  # used for saving
import numpy as np
import traceback
from tqdm import tqdm
from dandi.dandiapi import DandiAPIClient

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


known_implementations.keys()

 
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

        self.behave = None
        self.behave_num = 0

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """ Generates an array that serves as an initial source of data.

        Initial array is a 100 row, 5 column numpy matrix that contains
        integers from 1-99, inclusive. 
        """

        logger.info('Beginning setup for Generator')


        # set logging level to ERROR to avoid printing out a lot of messages
        fsspec_logger = logging.getLogger("fsspec")
        fsspec_logger.setLevel(logging.ERROR)

        # need to do "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" in terminal if you get an error
        dandiset_id = '000054'  # can change according to different dandisets
        filepath = 'sub-F1/sub-F1_ses-20190407T210000_behavior+ophys.nwb'
        
        with DandiAPIClient() as client:
            asset = client.get_dandiset(dandiset_id, '0.210819.1547').get_asset_by_path(filepath)
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        logger.info('Got s3 urls')

        # first, create a virtual filesystem based on the http protocol and use
        # caching to save accessed data to RAM.

        logger.info('Beginning creating virtual filesystem')
        try:
            fs=fsspec.filesystem("http")
            # HTTPFileSystem requires "requests" and "aiohttp" to be installed
        except Exception as e:
            logger.error(f"-------------------File System Exception: {e}")
            logger.error(traceback.format_exc())

        logger.info('Completed creating virtual filesystem')

        # next, open the file
        f = fs.open(s3_url, "rb")
        file = h5py.File(f)
        self.io = pynwb.NWBHDF5IO(file=file, mode="r")
        self.nwbfile = self.io.read()
        try:
            data = self.nwbfile.acquisition['TwoPhotonSeries'].data[0:10,::]
        except Exception as e:
            logger.error(
                "Error occurred while loading data:", e)
        self.data = data

        try:
            behave = self.nwbfile.processing['behavior']['BehavioralTimeSeries']['pos'].data[0:10]
        except Exception as e:
            logger.error(
                "Error occurred while loading behavior:", e)
        self.behave = behave
        logger.info('Completed setup for Generator')

    def stop(self):
        """ Save current randint vector to a file.
        """

        logger.info("Generator stopping")
        np.save("sample_generator_data.npy", self.data)
        return 0

    def runStep(self):
        """ Generates additional data after initial setup data is exhausted.

        Data is of a different form as the setup data in that although it is 
        the same size (5x1 vector), it is uniformly distributed in [1, 10] 
        instead of in [1, 100]. Therefore, the average over time should 
        converge to 5.5. 
        """

        if (self.frame_num < np.shape(self.data)[0]):
            data_id = self.client.put(
                self.data[self.frame_num], str(f"Gen_raw: {self.frame_num}"))
            # logger.info('Put data in store')
            try:
                self.q_out.put([[data_id, str(self.frame_num)]])
                self.frame_num += 1
            except Exception as e:
                logger.error(
                    f"--------------------------------Generator Exception: {e}")
        else:
            if (self.frame_num < len(self.nwbfile.acquisition['TwoPhotonSeries'].data)):
                try:
                    start_index = self.frame_num
                    end_index = start_index + 10
                    sample_data = self.nwbfile.acquisition['TwoPhotonSeries'].data[start_index:end_index, ::]
                except Exception as e:
                    logger.error("Error occurred while loading data:", e)
                self.data = np.concatenate((self.data, sample_data), axis=0)
            else:
                logger.warning("All available data have been processed")

        if (self.behave_num < np.shape(self.behave)[0]):
            behave_id = self.client.put(
                self.behave[self.behave_num], str(f"Gen_raw: {self.behave_num}"))
            # logger.info('Put data in store')
            try:
                self.links['bq_out'].put([[behave_id, str(self.behave_num)]])
                self.behave_num += 1
            except Exception as e:
                logger.error(
                    f"--------------------------------Generator Exception: {e}")
        else:
            if (self.behave_num < len(self.nwbfile.processing['behavior']['BehavioralTimeSeries']['speed'].data)):
                try:
                    start_index = self.behave_num
                    end_index = start_index + 10
                    sample_behave = self.nwbfile.processing['behavior']['BehavioralTimeSeries']['speed'].data[start_index:end_index]
                except Exception as e:
                    logger.error("Error occurred while loading data:", e)
                self.behave = np.concatenate((self.behave, sample_behave), axis=0)
            else:
                logger.info("Completed running whole NWB file")
        