from improv.actor import Actor, RunManager
import time
import numpy as np
import mat73
import scipy.signal as signal
from sklearn import random_projection as rp
from proSVD.proSVD import proSVD
from queue import Empty
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DimReduction(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Load initial data from Acquirer, performs initalization and send to bubblewrap"""
        init_id = None
        while init_id is None:
            try:
                init_id = self.q_in.get(timeout = 0.0005)
            except Empty: pass
        logger.info("Got init data")
        my_list = self.client.getID(init_id)
        dat_shape_0 = my_list[0]
        dat_init = np.array(my_list[1])

        bw_id = self.client.put(dat_shape_0, "dat_shape_bw")
        self.q_out.put(bw_id)

        # proSVD params
        k = 6 # reduced dimension
        trueSVD = True # whether proSVD should track true SVD basis (a little slower)
        l = 1 # columns per update
        l1 = 100 # columns used to initialize
        decay = 1 # 1 = effective window is all of data
        bin_size_ms = 10
        M = 20

        # smoothing params
        kern_sd = 50
        self.smooth_filt = signal.gaussian(int(6 * kern_sd / bin_size_ms), int(kern_sd / bin_size_ms), sym=True)
        self.smooth_filt /=  np.sum(self.smooth_filt)
        data_init_smooth = np.apply_along_axis(lambda x, filt: np.convolve(x, filt, 'same'), 
                                            0, dat_init, filt=self.smooth_filt)
        
        # initialize proSVD
        self.pro = proSVD(k=k, decay_alpha=decay, trueSVD=trueSVD, history=0)
        self.pro.initialize(data_init_smooth.T)
        # storing dimension-reduced data
        self.data_red = np.zeros((dat_shape_0, k))
        self.data_red[:l1, :] = data_init_smooth @ self.pro.Q
        bw_id = self.client.put(self.data_red[:M], "bw_data")
        #send to bubblewrap
        self.q_out.put(bw_id)
        self.pro_diffs = []
        self.smooth_window = dat_init[l1-len(self.smooth_filt):l1, :]

    def runStep(self):
        """update proSVD at each step using data from Acquirer and send to bubblewrap"""
        try:
            res = self.q_in.get(timeout=0.0005)
            data_curr = self.client.getID(res[1])[1]
            self.t = self.client.getID(res[1])[0]
            start, end = self.t, self.t+self.pro.w_len
            self.smooth_window[:-1, :] = self.smooth_window[1:, :]
            self.smooth_window[-1, :] = data_curr
            dat_smooth = self.smooth_filt @ self.smooth_window
            # update proSVD
            self.pro.preupdate()
            self.pro.updateSVD(dat_smooth[:, None])
            self.pro.postupdate()
            self.pro_diffs.append(np.linalg.norm(self.pro.Q-self.pro.Q_prev, axis=0))

            self.data_red[start:end, :] = dat_smooth @ self.pro.Q
            # send to bubblewrap
            try:
                id = self.client.put(self.data_red[self.t], "dim_bubble" + str(self.t))
                self.q_out.put([int(self.t), id])
                self.links['v_out'].put([int(self.t), id])
            except Exception as e:
                logger.error("Dimension reduction general exception: {}".format(e))
                logger.error(traceback.format_exc())
        except Empty:
            return None

