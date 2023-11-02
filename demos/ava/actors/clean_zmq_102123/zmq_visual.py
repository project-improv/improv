import os
from queue import Empty
import time
import traceback
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets

from improv.actor import Actor
from improv.store import CannotGetObjectError, ObjectNotFoundError

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass


class AVAVisual(Actor):
    """Visual as a single Actor — saving out audio and plots.
    TODO GUI for streaming audio, specs, latents to embedding.
    """

    def __init__(self, *args, out_path=None, umap_dict_fname=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

        self.out_path = out_path
        self.umap_dict = umap_dict_fname

    def setup(self):
        """Setup for AVA Visual.
        """
        logger.info(f"Running setup for {self.name}.")        
        # self.audio_dir = os.path.join(self.out_path, 'audio')
        # self.spec_dir = os.path.join(self.out_path, 'spec')
        # self.proj_dir = os.path.join(self.out_path, 'proj')
        self.latents_dir = os.path.join(self.out_path, 'latents')
        self.plots_dir = os.path.join(self.out_path, 'plots')

        # os.makedirs(self.audio_dir, exist_ok=True)
        # os.makedirs(self.spec_dir, exist_ok=True)
        # os.makedirs(self.proj_dir, exist_ok=True)

        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.latents_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        umap_dict = joblib.load(self.umap_dict)

        self.transform = umap_dict['transform']
        self.embedding = umap_dict['embed']

        self.fig, self.ax = plt.subplots()

        # From bubblewrap (CaIman Visual Actor)
        #self.visual is CaimanVisual
        # self.visual = visual
        # self.visual.setup()
        
        self.dropped_wav = []

        self.viz_timestamps = []
        self.viz_total_times = []

        self.seg_num = 0

        logger.info(f"Completed setup for {self.name}.")

    def stop(self):
        """Stop procedure...
        """
        logger.info(f"{self.name} stopping.")
    
        print('Visual broke, avg time per segment:', np.mean(self.viz_total_times))
        print('Visual got through', self.seg_num, ' segments')

    # def run(self):
        """_summary_
        """
        # logger.info('Loading FrontEnd')
        # self.app = QtWidgets.QApplication([])
        # self.viewer = FrontEnd(self.visual, self.q_comm, self.q_sig)
        # self.viewer.show()
        # logger.info('GUI ready')
        # self.q_comm.put([Spike.ready()])
        # self.visual.q_comm.put([Spike.ready()])
        # self.app.exec_()
        # logger.info('Done running GUI')

    def runStep(self):
        """_summary_
        """
        if self.done:
            pass

        ids = self._checkInput()

        if ids is not None:
            self.done = False
            try:
                latents = self.client.getID(ids[0])

                np.save(os.path.join(self.latents_dir, str(self.seg_num) + '.npy'), latents)

                xmin, xmax, ymin, ymax = self.set_ax_limits(self.embedding)

                with warnings.catch_warnings():
                    try:
                        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
                    except NameError:
                        pass
                    embedding = self.transform.transform(latents)

                self.plot_map(embedding, xmin, xmax, ymin, ymax, marker_s=10)

            except ObjectNotFoundError:
                logger.error(f"{self.name}: Latent means {self.seg_num} unavailable from store, dropping")
                # self.dropped_wav.append(self.seg_num)
            except KeyError as e:
                logger.error(f"{self.name}: Key error... {e}")
                # self.dropped_wav.append(self.seg_num)
            except Exception as e:
                logger.error(f"{self.name} error: {type(e).__name__}: {e} during segment number {self.seg_num}")
                logger.info(traceback.format_exc())
                # self.dropped_wav.append(self.seg_num)

        else:
            pass

        self.seg_num += 1

        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    def _checkInput(self):
        """Check to see if we have latent means, q_in.

        Returns:
            _type_: _description_
        """
        try:
            res = self.q_in.get(timeout=0.005)
            return res
        #TODO: additional error handling
        except Empty:
            pass
            # logger.info('No .wav files for processing')
            # return None

    def set_ax_limits(self, original_embed):
        """_summary_

        Args:
            original_embed (_type_): _description_

        Returns:
            _type_: _description_
        """
	    # Calculate x and y limits.
        xmin = np.min(original_embed[:,0])
        ymin = np.min(original_embed[:,1])
        xmax = np.max(original_embed[:,0])
        ymax = np.max(original_embed[:,1])
        x_pad = 0.05 * (xmax 
        - xmin)
        y_pad = 0.05 * (ymax - ymin)
        xmin, xmax = xmin - x_pad, xmax + x_pad
        ymin, ymax = ymin - y_pad, ymax + y_pad

        return xmin, xmax, ymin, ymax

    def plot_map(self, new_embed, xmin=None, xmax=None, ymin=None, ymax=None, marker_c='black', marker_s=50.0, marker_marker='o'):
        """_summary_

        Args:
            new_embed (_type_): _description_
            xmin (_type_, optional): _description_. Defaults to None.
            xmax (_type_, optional): _description_. Defaults to None.
            ymin (_type_, optional): _description_. Defaults to None.
            ymax (_type_, optional): _description_. Defaults to None.
            marker_c (str, optional): _description_. Defaults to 'black'.
            marker_s (float, optional): _description_. Defaults to 50.0.
            marker_marker (str, optional): _description_. Defaults to 'o'.
        """
        
        self.ax.scatter(new_embed[0], new_embed[1], s=marker_s, marker=marker_marker, c=marker_c, rasterized=True)
        if xmin and xmax and ymin and ymax is not None:
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        plt.savefig(os.path.join(self.plots_dir, f"{self.seg_num}.svg"))
        plt.savefig(os.path.join(self.plots_dir, f"{self.seg_num}.pdf"))