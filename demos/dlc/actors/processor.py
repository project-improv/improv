import time
import pickle
import json
import cv2
import numpy as np
from improv.store import Store, CannotGetObjectError, ObjectNotFoundError
from os.path import expanduser
import os
from queue import Empty
from improv.actor import Actor, Spike, RunManager
import traceback
from dlclive import DLCLive, Processor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DLCProcessor(Actor):
    '''Wraps DLC-Live functionality to
       interface with our pipeline.
    '''
    def __init__(self, *args, video_path=None,  model_path=None, config_file=None):
        super().__init__(*args)
        logger.info(video_path, model_path)
        self.param_file = config_file
        self.video_path = video_path
        self.frame_number = 0
        if model_path is None:
            logger.error("Must specify a DLC live model path.")
        else:
            self.model_path = model_path # path to saved DLCLive model

    def setup(self):
        ''' Initialize DLC model
        '''
        logger.info('Running setup for '+self.name)
        self.done = False
        self.dropped_frames = []

        # initialize the DLCLive object with the specified model path
        self.dlc_live = DLCLive(
            model_path = self.model_path,
            processor = Processor()
        )

        #self.loadParams(param_file=self.param_file)
        #self.params = self.client.get('params_dict')

    def run(self):
        '''Run the processor continually on input frames
        '''

        self.timestamp = []
        self.counter = 0

        with RunManager(self.name, self.runProcess, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

        # before = self.params['init_batch']
        # nb = self.onAc.params.get('init', 'nb')
        # np.savetxt('raw_C.txt', np.array(self.onAc.estimates.C_on[nb:self.onAc.M, before:self.frame_number+before]))

        print('Number of times coords updated ', self.counter)

        # with open('../S.pk', 'wb') as f:
        #     init = self.params['init_batch']
        #     S = np.stack([osi.s[init:] for osi in self.onAc.estimates.OASISinstances])
        #     print('--------Final S shape: ', S.shape)
        #     pickle.dump(S, f)
        # with open('../A.pk', 'wb') as f:
        #     nb = self.onAc.params.get('init', 'nb')
        #     A = self.onAc.estimates.Ab[:, nb:]
        #     print(type(A))
        #     pickle.dump(A, f)


    def runProcess(self):
        ''' Run process. Runs once per frame.
            Output is a location in the DS to continually
            place the Estimates results, with ref number that
            corresponds to the frame number
        '''
        init = self.params['init_batch']
        frame = self._checkFrames()

        if frame is not None:
            t = time.time()
            self.done = False
            try:
                self.frame = self.client.getID(frame[0][str(self.frame_number)])
                self.frame = self._processFrame(self.frame, self.frame_number+init)
                t2 = time.time()
                self._fitFrame(self.frame_number+init, self.frame.reshape(-1, order='F'))
                self.fitframe_time.append([time.time()-t2])
                self.putEstimates()
                self.timestamp.append([time.time(), self.frame_number])
            except ObjectNotFoundError:
                logger.error('Processor: Frame {} unavailable from store, droppping'.format(self.frame_number))
                self.dropped_frames.append(self.frame_number)
                self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_frames.append(self.frame_number)
            except Exception as e:
                logger.error('Processor error: {}: {} during frame number {}'.format(type(e).__name__,
                                                                                            e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_frames.append(self.frame_number)
            self.frame_number += 1
            self.total_times.append(time.time()-t)
        else:
            pass

    def loadParams(self, param_file=None):
        ''' Load parameters from file or 'defaults' into store
            TODO: accept user input from GUI
        '''
        cwd = os.getcwd()+'/'
        if param_file is not None:
            try:
                params_dict = self._load_params_from_file(param_file)
                params_dict['fnames'] = [cwd+self.init_filename]
            except Exception as e:
                logger.exception('File cannot be loaded. {0}'.format(e))
        else:
            # defaults from demo scripts;
            params_dict = {
                'fnames': [cwd+self.init_filename],
                'fr': 2,
                'model': ''
            }
        self.client.put(params_dict, 'params_dict')

    def _load_params_from_file(self, param_file):
        '''Filehandler for loading caiman parameters
            TODO: Error handling, check params as loaded
        '''
        params_dict = json.load(open(param_file, 'r'))
        return params_dict

    def putEstimates(self):
        ''' Put whatever estimates we currently have
            into the data store
        '''
        return


    def _checkFrames(self):
        ''' Check to see if we have frames for processing
        '''
        try:
            res = self.q_in.get(timeout=0.0005)
            return res
        #TODO: add'l error handling
        except Empty:
            # logger.info('No frames for processing')
            return None


    def _processFrame(self, frame, frame_number):
        ''' Do some basic processing on a single frame
        '''
        print(frame,frame_number)
        logger.log(frame)
        return frame


    def _fitFrame(self, frame_number, frame):
        ''' Fit the current frame
        '''
        try:
            '''call DLC function'''
            self.dlc_live.init_inference(frame)
            pose = self.dlc_live.get_pose(frame)
            print(pose)

        except Exception as e:
            logger.error('Fit frame error! {}: {}'.format(type(e).__name__, e))
            raise Exception


    def makeImage(self):
        '''Create image data for visualiation
        '''
        image = None

        return image

class NaNFrameException(Exception):
    pass
