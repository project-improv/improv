from improv.actor import Actor, RunManager
import time
import numpy as np
from scipy import io as sio
from sklearn import random_projection as rp
from proSVD import proSVD
from queue import Empty
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DimReduction(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.dataloc = './'
        self.file = 'WaksmanwithFaces_KS2.mat'
        self.shape = 15000
        matdict = sio.loadmat(self.dataloc + self.file, squeeze_me=True)
        spks = matdict['stall'][:, :10]

        k = 10
        l1 = k
        self.l = 10
        rp_dim = 200

        num_iters = np.ceil((self.shape - l1) / self.l).astype('int')

        self.pro_proj = np.zeros((k, self.shape))
        self.pro = proSVD.proSVD(k, history=num_iters, trueSVD=True)
        self.pro.initialize(spks)
        self.pro_proj[:, :l1] = self.pro.Q.T @ spks

        self.proj_stream_ssSVD = np.zeros(
            (self.pro.Qs.shape[1], self.pro.Qs.shape[2] * self.l)
        )
        proj_stream_SVD = np.zeros(
            (self.pro.Us.shape[1], self.pro.Us.shape[2] * self.l)
        )
        self.projs = [self.proj_stream_ssSVD, proj_stream_SVD]
        self.bases = [self.pro.Qs, self.pro.Us]

        self.transformer = rp.SparseRandomProjection(n_components=rp_dim)

        # X_rp = self.reduce_sparseRP(spks, transformer=self.transformer)
        self.pro.updateSVD(spks)

        for i, basis in enumerate(self.bases):
            curr_basis = basis[:, :, i]  # has first k components
            curr_neural = spks
            self.projs[i][:, :l1] = curr_basis.T @ curr_neural

    def run(self):
        print("Starting receiver loop ...")
        print("run proSVD")
        with RunManager(
            self.name, self.runProcess, self.setup, self.q_sig, self.q_comm
        ) as rm:
            print(rm)

    def runProcess(self):
        self.runProSVD()
        return

    def runProSVD(self):
        try:
            res = self.q_in.get(timeout=0.0005)
            self.spks = self.client.getID(res[1])[1]
            self.t = self.client.getID(res[1])[0]

            # X_rp = self.reduce_sparseRP(self.spks, transformer=self.transformer)
            self.pro.updateSVD(self.spks)

            for i, basis in enumerate(self.bases):
                curr_basis = basis[:, :, i]  # has first k components
                # aligning neural to Q (depends on l1 and l)
                curr_neural = self.spks
                # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
                self.projs[i][:, self.t : self.t + self.l] = curr_basis.T @ curr_neural
                id = self.client.put(
                    [self.t, self.projs[i][:, self.t : self.t + self.l]],
                    'dim_bubble' + str(self.t),
                )
                try:
                    self.q_out.put([str(self.t), id])
                except Exception as e:
                    logger.error('Acquirer general exception: {}'.format(e))

        except Empty:
            return None

    def reduce_sparseRP(X, comps=100, eps=0.1, transformer=None):
        np.random.seed(42)
        if transformer is None:
            transformer = rp.SparseRandomProjection(n_components=comps, eps=eps)
        X_new = transformer.fit_transform(X)
        return X_new
