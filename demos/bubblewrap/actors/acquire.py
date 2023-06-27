from improv.actor import Actor, RunManager, Signal
import os
import numpy as np
import mat73
import time
import logging
import traceback
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Acquirer(Actor):
    def __init__(self, *args, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not filename: logger.error('Error: Filename not specified')
        self.file = filename
        self.frame_num = 0
        self.done = False
        self.framerate = 24
        self.timestamp = []
        self.total_times = []

    def setup(self):
        # get unsorted vs sorted units
        data_dict = mat73.loadmat(self.file)
        units_unsorted = []
        units_sorted = []
        for ch_curr in data_dict['spikes']:
            units_unsorted.append(ch_curr[0]) # from data description, first unit is unsorted
            for unit_curr in ch_curr[1:]:
                if unit_curr is not None:
                    units_sorted.append(unit_curr)
        
        # getting binned spikes
        bin_size_ms = 10
        # window for binning
        mintime, maxtime = 100, 0
        for spk_times_curr in units_sorted:
            mintime = min(spk_times_curr[0], mintime)
            maxtime = max(spk_times_curr[-1], maxtime)
        # get binned spikes
        spk_bins = np.arange(mintime, maxtime, bin_size_ms/1000)
        spks_binned = []
        for i, unit_curr in enumerate(units_sorted):
            spks_binned_curr, _ = np.histogram(unit_curr, bins=spk_bins, range=(spk_bins[0], spk_bins[-1]))
            spks_binned.append(spks_binned_curr)
        self.data = np.array(spks_binned).T

        self.l = 1 # columns per update
        l1 = 100 # columns used to initialize
        self.t = l1
        self.num_iters = np.floor((self.data.shape[0] - l1 - self.l)/self.l).astype('int')


        init_id = self.client.put([self.data.shape[0], self.data[:l1, :]], "init_data")
        logger.info("Putted init data")
        self.q_out.put(init_id)

    def stop(self):
        logger.info(f"Stopped running Acquire, avg time per frame: {np.mean(self.total_times)}")
        logger.info(f"Acquire got through {self.frame_num} frames")
    
    def runStep(self):
        if self.done:
            pass
        elif self.frame_num < self.num_iters:
            start, end = self.t, self.t + 1
            frame = self.data[start:end, :]
            t = time.time()
            id = self.client.put([self.t, frame], "acq_bubble" + str(self.frame_num))
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.q_out.put([str(self.frame_num), id])
                self.frame_num += 1
                self.t += self.l
                # also log to disk #TODO: spawn separate process here?
            except Exception as e:
                logger.error("Acquirer general exception: {}".format(e))
                logger.error(traceback.format_exc())


            time.sleep(1/self.framerate)  # pretend framerate
            self.total_times.append(time.time() - t)

        else:  # simulating a done signal from the source
            logger.error("Done with all available frames: {0}".format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get e.g. a shutdown signal
