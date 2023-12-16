
import numpy as np
from queue import Empty
from bubblewrap import Bubblewrap
from improv.actor import Actor
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Bubble(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def setup(self):
        """Load data from dim reduction and perform node initalization"""
        shape_id = None
        while shape_id is None:
            try:
                shape_id = self.q_in.get(timeout=0.0005)
            except Empty: pass
        dat_shape_0 = self.client.get(shape_id)
        # init bubblewrap
        M = 20
        N = 50
        lam = 1e-3
        nu = 1e-3
        grads_per_obs = 1
        step = 8e-3
        k = 6 # reduced dimension
        d = k
        num_d = k
        T = dat_shape_0 - M # iters
        # sigma_scale = 1e3
        B_thresh = -10
        n_thresh = 5e-4
        eps = 0
        P = 0 
        t_wait = 1 # breadcrumbing
        future_distance = 5
        self.bw = Bubblewrap(N, num_d, d, step=step, lam=lam, eps=eps, M=M, nu=nu, 
                        t_wait=t_wait, n_thresh=n_thresh, B_thresh=B_thresh)
        id = None
        while id is None:
            try:
                id = self.q_in.get(timeout = 0.0005)
            except Empty: pass
        init_data = self.client.getID(id)

        for i in np.arange(0, M):
            self.bw.observe(init_data[i])
        self.bw.init_nodes()
        logger.info("Nodes initialized")

        self._getStoreInterface()

    def runStep(self):
        """Observe new data from dim reduction and update bubblewrap"""
        try:
            ids = self.q_in.get(timeout=0.0005)

            # expect ids of size 2 containing data location and frame number
            new_data = self.client.getID(ids[1])
            self.frame_number = ids[0]

            self.bw.observe(new_data)
            self.bw.e_step()
            self.bw.grad_Q()
            self.putOutput()
        except Empty:
            pass

    def putOutput(self):
        """Function for putting updated results into the store"""
        ids = []
        ids.append(self.client.put(np.array(self.bw.A), "A" + str(self.frame_number)))
        ids.append(self.client.put(np.array(self.bw.L), "L" + str(self.frame_number)))
        ids.append(self.client.put(np.array(self.bw.mu), "mu" + str(self.frame_number)))
        ids.append(self.client.put(
                    np.array(self.bw.n_obs), "n_obs" + str(self.frame_number)))
        ids.append(self.client.put(
                    np.array(self.bw.pred), "pred" + str(self.frame_number)))
        ids.append(self.client.put(
                    np.array(self.bw.entropy_list), "entropy" + str(self.frame_number)))
        ids.append(self.client.put(
                    np.array(self.bw.dead_nodes), "dead_nodes" + str(self.frame_number)))
        self.q_out.put([self.frame_number, ids])
