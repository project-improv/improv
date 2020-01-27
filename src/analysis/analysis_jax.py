import logging
import time
from queue import Empty
from typing import Dict

import cv2
import numpy as np

from nexus.actor import Actor, RunManager
from nexus.store import ObjectNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelAnalysisJax(Actor):

    def __init__(self, *args, params=None, theta=None, n_stim=21, optimizer=None, use_gpu=False):
        """
        # TODO Docs.
        :param optimizer: Choose an optimizer from jax.experimental.optimizers. Defaults to adagrad with 1e-5 learning rate.
        """
        super().__init__(*args)

        self.model_params = self._generate_model_params() if params is None else params
        self.model_theta = theta
        self.model_optimizer = optimizer
        self.model_use_gpu = use_gpu
        self.model = None  # Need to initialize JAX after setup due to interpreter fork.
        self.model_lls = list()

        self.n_stim = n_stim
        self.n_frame = 0
        self.window_size = 100

        self.stim_proc = StimProcessor(self.window_size, self.n_stim)
        self.color_plotter = ColorImagePlotter()

        self.ts_put = list()
        self.ts_total = list()
        self.ts_stim = list()

    def setup(self):
        from .sim_GLM_jax import simGLM
        # Model initialization with JIT compilation.
        self.model = simGLM(self.model_params, self.model_theta, self.model_optimizer, self.model_use_gpu)
        y = np.zeros((1, 1))
        s = np.zeros((self.model_params['ds'], 1))
        self.model.fit(y, s)
        logger.info('JAX ok')

    def run(self):
        with RunManager(self.name, self.run_analysis, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

    def run_analysis(self):
        t0 = time.time()

        # Get new stim data.
        lastOnOff, currStim, currStimID, allStims = self.update_stim()
        self.ts_stim.append(t0 - time.time())

        # Get data from Processor.
        Call, Cpop = None, None
        try:
            self.n_frame, coords_dict, img, S = self.update_processor()
        except ObjectNotFoundError:
            logger.error('Estimates unavailable from store, dropping.')
        except Empty:
            pass

        else:
            # Tuning curve.
            C = S
            estsAvg = self.stim_proc.stimAvg_start(S)
            globalAvg = np.mean(estsAvg[:, :8], axis=0)
            tune = [estsAvg[:, :8], globalAvg]

            # Colored image.
            color = self.color_plotter.plotColorFrame(img, coords_dict, estsAvg)

            # Model fitting.
            if self.n_frame > 0:  # First S is blank.

                ll = self.fit_model(S, currStimID)
                self.model_lls.append(ll)

                Cx = np.arange(self.n_frame - self.window_size, self.n_frame) \
                    if self.n_frame >= self.window_size else np.arange(0, self.n_frame)

                if C.shape[1] > 0:
                    Cpop = np.nanmean(C, axis=0)
                    Call = C  # already a windowed version #[:,self.n_frame-window:self.n_frame]

                n_neu = S.shape[0]

                self.put_analysis(Cx, Call, Cpop, tune, color, coords_dict, allStims,
                                  np.array(self.model.θ['w'][:n_neu, :n_neu]), self.model_lls)

        self.ts_total.append(time.time() - t0)

    def update_stim(self):
        """ Send new stim data and retrieve relevant data to/from StimProc ."""
        s_proc = self.stim_proc
        try:
            stim_raw = self.links['input_stim_queue'].get(timeout=0.001)
        except Empty:
            pass  # No change in input stimulus.
        else:
            s_proc.updateStim_start(stim_raw)

        return s_proc.lastOnOff, s_proc.currStim, s_proc.currStimID, s_proc.allStims

    def update_processor(self):
        ids_store = self.q_in.get(timeout=0.001)
        if ids_store is not None and ids_store[0] == 1:
            print('Analysis: missing frame')
            self.q_out.put([1])
            raise Empty

        n_frame = ids_store[-1]
        coords_dict, img, S = self.client.getList(ids_store[:-1])
        return n_frame, coords_dict, img, S

    def fit_model(self, S, stim_id):
        # Create a window.
        if self.n_frame < self.window_size:
            y_step = S[:, :self.n_frame]
            stim_step = stim_id[:, :self.n_frame]
        else:
            y_step = S[:, self.n_frame - self.window_size:self.n_frame]
            stim_step = stim_id[:, self.n_frame - self.window_size:self.n_frame]

        y_step = np.where(np.isnan(y_step), 0, y_step)

        return self.model.fit(y_step, stim_step)  # Return likelihood.

    @staticmethod
    def _generate_model_params() -> Dict:
        N = 100
        M = 100
        dh = 10
        ds = 8
        return {'dh': dh, 'ds': ds, 'dt': 0.1, 'n': 0, 'N_lim': N, 'M_lim': M}

    def put_analysis(self, Cx, Call, Cpop, tune, color, coordDict, allStims, w, LL):
        t0 = time.time()
        ids_store = [
            self.client.put(Cx, f'Cx{self.n_frame}'),
            self.client.put(Call, f'Call{self.n_frame}'),
            self.client.put(Cpop, f'Cpop{self.n_frame}'),
            self.client.put(tune, f'tune{self.n_frame}'),
            self.client.put(color, f'color{self.n_frame}'),
            self.client.put(coordDict, f'analys_coords{self.n_frame}'),
            self.client.put(allStims, f'stim{self.n_frame}'),
            self.client.put(w, f'w{self.n_frame}'),
            self.client.put(np.array(LL), f'LL{self.n_frame}'),
            self.n_frame
        ]
        self.q_out.put(ids_store)
        self.ts_put.append(time.time() - t0)


class ColorImagePlotter:
    mat_weight = np.array([
        [1, 0.25, 0],
        [0.75, 1, 0],
        [0, 2, 0],
        [0, 0.75, 1],
        [0, 0.25, 1],
        [0.25, 0, 1.],
        [1, 0, 1],
        [1, 0, 0.25],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0]
    ])

    def plotColorFrame(self, image, coords_dict, estsAvg):
        ''' Computes colored nicer background+components frame
        '''
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[..., 3] = 255
        # color = self.color.copy() #TODO: don't stack image each time?
        coords = [o['coordinates'] for o in coords_dict]
        if coords is not None:
            for i, c in enumerate(coords):
                # c = np.array(c)
                ind = c[~np.isnan(c).any(axis=1)].astype(int)
                # TODO: Compute all colors simultaneously! then index in...
                cv2.fillConvexPoly(color, ind, self._tuningColor(i, color[ind[:, 1], ind[:, 0]], estsAvg))

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly. 

        return color

    def _tuningColor(self, ind, inten, estsAvg):
        ''' ind identifies the neuron by number
        '''
        if estsAvg[ind] is not None:  # and np.sum(np.abs(self.estsAvg[ind]))>2:
            try:
                return self.manual_Color_Sum(estsAvg[ind])
            except ValueError:
                return (255, 255, 255, 0)
            except Exception:
                print('inten is ', inten)
                print('ests[i] is ', estsAvg[ind])
        else:
            return (255, 255, 255, 50)  # 0)

    def manual_Color_Sum(self, x):
        ''' x should be length 12 array for coloring
        '''

        color = x @ self.mat_weight

        blend = 0.8  # 0.3
        thresh = 0.2  # 0.1
        thresh_max = blend * np.max(color)

        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)

        if color.any() and np.linalg.norm(color - np.ones(3)) > 0.35:
            # print('color is ', color, 'with distance', np.linalg.norm(color-np.ones(3)))
            color *= 255
            return (color[0], color[1], color[2], 255)
        else:
            return (255, 255, 255, 10)


class StimProcessor:
    def __init__(self, window_size, n_stim):
        self.n_stim = n_stim
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = dict()
        self.stimStart = dict()
        self.window_size = window_size

        self.stimID_dict = {3: 0, 10: 1, 9: 2, 16: 3, 4: 4, 14: 5, 13: 6, 12: 7}

        self.lastOnOff = None
        self.recentStim = [0] * self.window_size
        self.currStimID = np.zeros((8, 100000))  # FIXME
        self.currStim = -10
        self.allStims = {}

    def updateStim_start(self, stim):
        frame = list(stim.keys())[0]
        whichStim = stim[frame][0]
        stimID = self.stimID_dict.get(int(whichStim), -10)  # Defaults to -10.
        # print('Got ', frame, whichStim, stim[frame][1])
        if whichStim not in self.stimStart.keys():
            self.stimStart.update({whichStim: []})
            self.allStims.update({stimID: []})
        if abs(stim[frame][1]) > 1:
            curStim = 1  # on
            self.allStims[stimID].append(frame)
        else:
            curStim = 0  # off
        if self.lastOnOff is None:
            self.lastOnOff = curStim
        elif self.lastOnOff == 0 and curStim == 1:  # was off, now on
            self.stimStart[whichStim].append(frame)
            print('Stim ', whichStim, ' started at ', frame)
            if stimID > -10:
                self.currStimID[stimID, frame] = 1
            self.currStim = stimID
        else:
            # stim off
            self.currStimID[:, frame] = np.zeros(8)

        self.lastOnOff = curStim

    def stimAvg_start(self, S):
        ests = self.S = S  # ests = self.C
        ests_num = ests.shape[1]

        polarAvg = [np.zeros(ests.shape[0])] * 12
        estsAvg = [np.zeros(ests.shape[0])] * self.n_stim
        for s, l in self.stimStart.items():
            l = np.array(l)
            # print('stim ', s, ' is l ', l)
            if l.size > 0:
                onInd = np.array([np.arange(o + 4, o + 18) for o in np.nditer(l)]).flatten()
                onInd = onInd[onInd < ests_num]
                # print('on ', onInd)
                offInd = np.array([np.arange(o - 10, o + 25) for o in np.nditer(l)]).flatten()  # TODO replace
                offInd = offInd[offInd >= 0]
                offInd = offInd[offInd < ests_num]
                # print('off ', offInd)
                try:
                    if onInd.size > 0:
                        onEst = np.mean(ests[:, onInd], axis=1)
                    else:
                        onEst = np.zeros(ests.shape[0])
                    if offInd.size > 0:
                        offEst = np.mean(ests[:, offInd], axis=1)
                    else:
                        offEst = np.zeros(ests.shape[0])
                    try:
                        estsAvg[int(s)] = onEst - offEst  # (onEst / offEst) - 1
                    except FloatingPointError:
                        print('Could not compute on/off: ', onEst, offEst)
                        estsAvg[int(s)] = onEst
                    except ZeroDivisionError:
                        estsAvg[int(s)] = np.zeros(ests.shape[0])
                except FloatingPointError:  # IndexError:
                    logger.error('Index error ')
                    print('int s is ', int(s))

        estsAvg = np.array(estsAvg)
        # TODO: efficient update, no copy
        polarAvg[2] = estsAvg[9, :]  # np.sum(estsAvg[[9,11,15],:], axis=0)
        polarAvg[1] = estsAvg[10, :]
        polarAvg[0] = estsAvg[3, :]  # np.sum(estsAvg[[3,5,8],:], axis=0)
        polarAvg[7] = estsAvg[12, :]
        polarAvg[6] = estsAvg[13, :]  # np.sum(estsAvg[[13,17,18],:], axis=0)
        polarAvg[5] = estsAvg[14, :]
        polarAvg[4] = estsAvg[4, :]  # np.sum(estsAvg[[4,6,7],:], axis=0)
        polarAvg[3] = estsAvg[16, :]

        # for color summation
        polarAvg[8] = estsAvg[5, :]
        polarAvg[9] = estsAvg[6, :]
        polarAvg[10] = estsAvg[7, :]
        polarAvg[11] = estsAvg[8, :]

        self.estsAvg = np.abs(np.transpose(np.array(polarAvg)))
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        # self.estsAvg = np.clip(self.estsAvg*4, 0, 4)

        return self.estsAvg
