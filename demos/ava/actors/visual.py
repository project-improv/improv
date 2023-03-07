import os
import time
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import numpy as np
from queue import Empty
import logging; logger = logging.getLogger(__name__)


from improv.actor import Actor, RunManager

class AVAVisual(Actor):
    ''' Visual as a single Actor — only saving out plots
    TODO GUI for streaming audio, specs, adding latents to embedding
    '''

    def __init__(self, *args, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_path = out_path

# From bubblewrap (CaIman Visual Actor)
    def setup(self):
	#self.visual is CaimanVisual
        # self.visual = visual
        # self.visual.setup()
        # logger.info('Running setup for '+self.name)
        self.audio_dir = os.path.join(self.out_path, 'audio')
        self.spec_dir = os.path.join(self.out_path, 'spec')
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.spec_dir, exist_ok=True)

    def run(self):

        self.viz_total_times = []

        with RunManager(self.name, self.runVisual, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        # logger.info('Loading FrontEnd')
        # self.app = QtWidgets.QApplication([])
        # self.viewer = FrontEnd(self.visual, self.q_comm, self.q_sig)
        # self.viewer.show()
        # logger.info('GUI ready')
        # self.q_comm.put([Spike.ready()])
        # self.visual.q_comm.put([Spike.ready()])
        # self.app.exec_()
        # logger.info('Done running GUI')

        print('Visual broke, avg time per segment:', np.mean(self.viz_total_times))
        print('Visual got through', self.seg_num, ' segments')

    def runVisual(self):

        t = time.time()

        try:
            self.getData()
            self.plotAudio()
            self.plotSpec()
        except Empty as e:
            pass
        except Exception as e:
            # logger.error('Visual: Exception in get data: {}'.format(e))
            pass

        self.viz_total_times.append((time.time() - t)*1000.0)

    def getData(self):
        try:
            audio = self.q_audio.get(timeout=0.005)
            spec_latents = self.q_in.get(timeout=0.005)
            self.audio = self.client.getID(audio[0])
            self.fname = self.client.getID(audio[1])
            self.spec = self.client.getID(spec_latents[0])
            self.latents = self.client.getID(spec_latents[1])
            self.seg_num = self.client.getID(spec_latents[2])
        except Empty as e:
            pass
        except Exception as e:
            # logger.error('Visual: Exception in get data: {}'.format(e))
            pass

    def plotAudio(self):
        try:
            plt.plot(self.audio)
            plt.imsave(os.path.join(self.audio_dir + self.fname + str(self.seg_num) + '.png'))
            plt.close()
        except Empty as e:
            pass
        except Exception as e:
            # logger.error('Visual: Exception in get data: {}'.format(e))
            pass

    def plotSpec(self):
        try:
            plt.imshow(self.spec)
            plt.imsave(os.path.join(self.spec_dir + str(self.fname) + str(self.seg_num) + '.png'))
            plt.close()
        except Empty as e:
            pass
        except Exception as e:
            # logger.error('Visual: Exception in get data: {}'.format(e))
            pass

    # def plotLatents(self):
    #     try:
    #         embedding = 
    #         plt.imshow(self.latent_embedding)
    #         plt.imsave(os.path.join(self.out_path, self.fname) + '_spec.png')
    #         plt.close()
    #     except Empty as e:
    #         pass
    #     except Exception as e:
    #         # logger.error('Visual: Exception in get data: {}'.format(e))
    #         pass
