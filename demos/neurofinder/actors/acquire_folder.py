import time
import os
import h5py
import struct
import numpy as np
import random

from pathlib import Path
from skimage.io import imread
from improv.actor import Actor, Spike, RunManager
from queue import Empty

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FolderAcquirer(Actor):
    """Current behavior is looping over all files in a folder.
    Class to read TIFF files in a specified {path} from disk.
    Designed for scenarios when new TIFF files are created during the run.
    Reads only new TIFF files (after Run started) and put on to the Plasma store.
    If there're multiple files, files are loaded by name.
    """

    def __init__(self, *args, folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.done = False
        self.path = Path(folder)

        self.frame_num = 0
        self.files = []

        if not self.path.exists() or not self.path.is_dir():
            raise AttributeError("Folder {} does not exist.".format(self.path))

    def setup(self):
        pass

    def saveImgs(self):
        self.imgs = []
        files = {f for f in self.path.iterdir() if f.suffix in [".tif", ".tiff"]}
        files = sorted(list(files))
        for file in files:
            img = self.get_tiff(file)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs)
        f = h5py.File("output/sample.h5", "w", libver="latest")
        f.create_dataset("default", data=self.imgs)
        f.close()

    def run(self):
        """Triggered at Run
        Get list of files in the folder and use that as the baseline.
        """
        self.total_times = []
        self.timestamp = []

        self.files = [f for f in self.path.iterdir() if f.suffix in [".tif", ".tiff"]]

        with RunManager(
            self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm
        ) as rm:
            print(rm)

        print("Acquire broke, avg time per frame: ", np.mean(self.total_times))
        print("Acquire got through ", self.frame_num, " frames")

    def runAcquirer(self):
        """Main loop. If there're new files, read and put into store."""
        t = time.time()
        try:
            obj_id = self.client.put(
                self.get_tiff(self.files[self.frame_num]),
                "acq_raw" + str(self.frame_num),
            )
            self.put([[obj_id, str(self.frame_num)]], save=[True])
            self.frame_num += 1
        except IndexError as e:
            pass

        self.total_times.append(time.time() - t)

    def get_tiff(self, file: Path):
        try:
            img = imread(file.as_posix())
        except ValueError as e:
            img = imread(file.as_posix())
            logger.error("File " + file.as_posix() + " had value error {}".format(e))
        return img  # [0,0,0, :, :,0]  #Extract first channel in this image set.
