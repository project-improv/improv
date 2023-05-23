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

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """ Generates an array that serves as an initial source of data.

        Initial array is a 100 row, 5 column numpy matrix that contains
        integers from 1-99, inclusive. 
        """

        logger.info('Beginning setup for Generator')
        # try:
        #     s3_urls = get_s3_urls_and_dandi_paths(dandiset_id="000054")
        #     # need to do "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" in terminal if you get an error
        # except Exception as e:
        #     logger.error(
        #         f"-------------------Get s3 urls Exception: {e}")
        #     logger.error(traceback.format_exc())

        dandiset_id = '000048'  # can change according to different dandisets
        filepath = 'sub-222549/sub-222549_ecephys+ophys.nwb'
        
        with DandiAPIClient() as client:
            asset = client.get_dandiset(
                dandiset_id, 'draft').get_asset_by_path(filepath)
            s3_url = asset.get_content_url(
                follow_redirects=1, strip_query=True)
        logger.info('Got s3 urls')

        # first, create a virtual filesystem based on the http protocol and use
        # caching to save accessed data to RAM.

        logger.info('Beginning creating virtual filesystem')
        try:
            fs = CachingFileSystem(
                fs=fsspec.filesystem("http"),
                cache_storage="nwb-cache",  # Local folder for the cache
                # HTTPFileSystem requires "requests" and "aiohttp" to be installed
            )
        except Exception as e:
            logger.error(f"-------------------File System Exception: {e}")
            logger.error(traceback.format_exc())

        logger.info('Completed creating virtual filesystem')
        # next, open the file
        self.data = []
        # set logging level to ERROR to avoid printing out a lot of messages
        fsspec_logger = logging.getLogger("fsspec")
        fsspec_logger.setLevel(logging.ERROR)

        # for s3_url in s3_urls:
        with fs.open(s3_url, "rb") as f:
            with h5py.File(f) as file:
                with pynwb.NWBHDF5IO(file=file, load_namespaces=True) as io:
                    nwbfile = io.read()
                    # collect all the data into a list named self.data
                    logger.info(type(nwbfile))
                    try:
                        data = nwbfile.acquisition['TwoPhotonSeries_green'].data[:]

                    except Exception as e:
                        logger.error(
                            "Error occurred while loading data:", e)
                    self.data = data
        # self.data = np.asmatrix(np.random.randint(100, size = (100, 5)))
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
        # logger.info(self.frame_num)
        # logger.info(self.data)

        if (self.frame_num < np.shape(self.data)[0]):
            data_id = self.client.put(
                self.data[self.frame_num], str(f"Gen_raw: {self.frame_num}"))
            # logger.info('Put data in store')
            try:
                self.q_out.put([[data_id, str(self.frame_num)]])
                logger.info('Sent message on')
                self.frame_num += 1
            except Exception as e:
                logger.error(
                    f"--------------------------------Generator Exception: {e}")
        else:
            # self.data = np.concatenate((self.data, np.asmatrix(
            #     np.random.randint(10, size=(1, 5)))), axis=0)
            pass
