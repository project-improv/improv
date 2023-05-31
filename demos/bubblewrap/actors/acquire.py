from improv.actor import Actor, RunManager, Spike
import os
import numpy as np
from scipy import io as sio
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Acquirer(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.done = False
        self.framerate = 1

    def setup(self):
        dataloc = "./"
        file = "WaksmanwithFaces_KS2.mat"
        matdict = sio.loadmat(dataloc + file, squeeze_me=True)
        spks = matdict["stall"]

        # truncate so it doesn't take forever
        self.arr = spks[:, :15000]
        self.l = 10  # num cols to process per iter
        k = 10  # num components to keep
        l1 = k
        self.t = l1
        self.num_iters = np.ceil((spks.shape[1] - l1) / self.l).astype("int")
        self.chunk_size = 10

    def run(self):
        print("Starting receiver loop ...")
        print("run acquire")
        self.timestamp = []
        self.total_times = []
        with RunManager(
            self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm
        ) as rm:
            print(rm)

        print("Done running Acquire, avg time per frame: ", np.mean(self.total_times))
        print("Acquire got through ", self.frame_num, " frames")

    def runAcquirer(self):
        if self.done:
            pass
        elif self.frame_num < self.num_iters:
            frame = self.arr[:, self.t : self.t + self.l]
            t = time.time()
            id = self.client.put([self.t, frame], "acq_bubble" + str(self.frame_num))
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.q_out.put([str(self.frame_num), id])
                self.frame_num += 1
                self.t += self.chunk_size
                # also log to disk #TODO: spawn separate process here?
            except Exception as e:
                logger.error("Acquirer general exception: {}".format(e))

            time.sleep(self.framerate)  # pretend framerate
            self.total_times.append(time.time() - t)

        else:  # simulating a done signal from the source
            logger.error("Done with all available frames: {0}".format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get e.g. a shutdown signal
