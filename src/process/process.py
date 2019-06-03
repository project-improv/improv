import time
import pickle
import cv2
import numpy as np
import scipy.sparse
from nexus.store import Limbo, CannotGetObjectError, ObjectNotFoundError
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.source_extraction.cnmf.online_cnmf import OnACID
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import motion_correct_iteration_fast, tile_and_correct
from caiman.utils.visualization import get_contours
import caiman as cm
from os.path import expanduser
import os
from queue import Empty
from nexus.module import Module, Spike

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Module):
    '''Abstract class for the processor component
       Needs to take an image from data store (DS)
       Needs to output spikes estimates over time
       Will likely change specifications in the future
    '''
    
    def putEstimates(self):
        # Update the DS with estimates
        raise NotImplementedError


class CaimanProcessor(Processor):
    '''Wraps CaImAn/OnACID functionality to
       interface with our pipeline.
       Uses code from caiman/source_extraction/cnmf/online_cnmf.py
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.ests = None #neural activity
        self.coords = None

    def loadParams(self, param_file=None):
        ''' Load parameters from file or 'defaults' into store
            TODO: accept user input from GUI
            This also effectively registers specific params
            that CaimanProcessor needs with Nexus
        '''
        #TODO: Convert this to Tweak and load from there
        if param_file is not None:
            try:
                params_dict = self._load_params_from_file(param_file)
            except Exception as e:
                logger.exception('File cannot be loaded. {0}'.format(e))
        else:
            # defaults from demo scripts; CNMFParams does not set
            # each parameter needed by default (TODO change that?)
            # TODO add parameter validation inside Tweak

            home = expanduser("~")
            cwd = os.getcwd()
            params_dict = {'fnames': [cwd+'/data/Tolias_mesoscope_1.hdf5', cwd+'/data/Tolias_mesoscope_2.hdf5'],
                   'fr': 15,
                   'decay_time': 0.5,
                   'gSig': (3,3),
                   'p': 1,
                   'min_SNR': 1,
                   'rval_thr': 0.9,
                   'ds_factor': 1,
                   'nb': 2,
                   'motion_correct': True,
                   'init_batch': 100,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': False,
                   'K': 10,
                   'epochs': 1,
                   'max_shifts_online': np.ceil(10).astype('int'),
                   'pw_rigid': False,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': False,
                   'update_freq': 500,
                   'minibatch_shape': 100,
                   'output': 'outputEstimates'}
        self.client.put(params_dict, 'params_dict')
        #TODO: return code    

    def _load_params_from_file(self, param_file):
        '''Filehandler for loading caiman parameters
            TODO
        '''
        return {}

    def setup(self, config_file = None):
        ''' Create OnACID object and initialize it
                (runs initialize online)
            limboClient is a client to the data store server
        '''
        logger.info('Running setup for '+self.name)
        self.done = False
        self.dropped_frames = []
        self.coords = None
        self.ests = None
        self.A = None

        self.loadParams()
        self.params = self.client.get('params_dict')
        
        # MUST include inital set of frames
        # TODO: Institute check here as requirement to Nexus
        
        self.opts = CNMFParams(params_dict=self.params)
        self.onAc = OnACID(params = self.opts)
        self.frame_number = 0 #self.params['init_batch']
        #TODO: Need to rewrite init online as well to receive individual frames.
        self.onAc.initialize_online()
        self.max_shifts_online = self.onAc.params.get('online', 'max_shifts_online')

        return self

    def run(self):
        '''Run the processor continually on input frames
        '''
        self.process_time = []
        self.putAnalysis_time = []
        self.procFrame_time = [] #aka t_motion
        self.detect_time = []
        self.shape_time = []
        self.flag = False

        total_times = []

        while True:
            t = time.time()
            if self.flag: #if we have received run signal
                try: 
                    self.runProcess()
                    # if self.done:
                    #     logger.info('Done running Process')
                    #     print('Dropped frames: ', self.dropped_frames)
                    #     print('Total number of dropped frames ', len(self.dropped_frames))
                    #     print('mean time per fit frame ', np.mean(self.process_time))
                    #     #return
                    total_times.append(time.time()-t)
                except Exception as e:
                    logger.exception('What happened: {0}'.format(e))
                    #break  
            try: 
                signal = self.q_sig.get(timeout=0.005)
                if signal == Spike.run(): 
                    self.flag = True
                    logger.warning('Received run signal, begin running process')
                elif signal == Spike.quit():
                    print('Total number of dropped frames ', len(self.dropped_frames))
                    print('mean time per fit frame ', np.mean(self.process_time))
                    print('mean time per process frame ', np.mean(self.procFrame_time))
                    print('mean time per put analysis ', np.mean(self.putAnalysis_time))
                    logger.warning('Received quit signal, aborting')
                    break
                elif signal == Spike.pause():
                    logger.warning('Received pause signal, pending...')
                    self.flag = False
                elif signal == Spike.resume(): #currently treat as same as run
                    logger.warning('Received resume signal, resuming')
                    self.flag = True
            except Empty as e:
                pass #no signal from Nexus
            
        print('Processor broke, avg time per frame: ', np.mean(total_times))
        print('Processor got through ', self.frame_number, ' frames')

    def runProcess(self):
        ''' Run process. Runs once per frame.
            Output is a location in the DS to continually
            place the Estimates results, with ref number that
            corresponds to the frame number (TODO)
        '''
        #TODO: Error handling for if these parameters don't work
            #should implement in Tweak (?) or getting too complicated for users..
        
        #proc_params = self.client.get('params_dict')
        output = self.params['output']
        init = self.params['init_batch']
        frame = self._checkFrames()
        
        if frame is not None:
            self.done = False
            try:
                self.frame = self.client.getID(frame[0][str(self.frame_number)])
                self.frame = self._processFrame(self.frame, self.frame_number+init)
                #self.frame = frame.copy()
                t = time.time()
                self._fitFrame(self.frame_number+init, self.frame.reshape(-1, order='F'))
                self.process_time.append([time.time()-t])
                #if frame_count % 5 == 0: 
                t = time.time()
                self.putEstimates(self.onAc.estimates, output) # currently every frame. User-specified?
                self.putAnalysis_time.append([time.time()-t])
            except ObjectNotFoundError:
                logger.error('Processor: Frame unavailable from store, droppping')
                self.dropped_frames.append(self.frame_number)
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_frames.append(self.frame_number)
            self.frame_number += 1
        else:
            pass
            # logger.error('Done with all available frames: {0}'.format(self.frame_number))
            # self.q_comm.put(None)
            # self.done = True

    def finalProcess(self, output):    
        if self.onAc.params.get('online', 'normalize'):
            # normalize final estimates for this set of files. Useful?
            #self._normAnalysis()
            self.putEstimates(self.onAc.estimates, output)
            #self._finalAnalysis(frame_number)
            #self.putEstimates(self.finalEstimates, output)
        
        #once runProcess has run once, need to reset init_batch to avoid internal errors
        #currently not logging this change; keeping all internal to caiman
        self.onAc.params.set('online', {'init_batch': 0})
        self.params['init_batch'] = 0
        # also need to extend dimensions of timesteps...TODO: better method init?
            #currently changed T1 inside online_cnmf....

        #TODO: determine a SINGLE location for params. Internal vs logged?
        #self.client.replace(self.params, 'params_dict')
        logger.info('Updated init batch after first run')

        self.shape_time = np.array(self.onAc.t_shapes)
        self.detect_time = np.array(self.onAc.t_detect)

        # np.savetxt('timing/process_time.txt', np.array(self.process_time))
        # np.savetxt('timing/shape_time.txt', self.shape_time)
        # np.savetxt('timing/detect_time.txt', self.detect_time)
        # np.savetxt('timing/putAnalysis_time.txt', np.array(self.putAnalysis_time))
        # np.savetxt('timing/procFrame_time.txt', np.array(self.procFrame_time))

        #print('mean time per frame ', np.mean(self.process_time))


    def putEstimates(self, estimates, output):
        ''' Put whatever estimates we currently have
            into the output location specified
            TODO rewrite output input
        '''
       
        nb = self.onAc.params.get('init', 'nb')
        A = self.onAc.estimates.Ab[:, nb:]
        #b = self.onAc.estimates.Ab[:, :nb] #toarray() ?
        C = self.onAc.estimates.C_on[nb:self.onAc.M, :self.frame_number]
        #f = self.onAc.estimates.C_on[:nb, :self.frame_number]
        
        #self.ests = C  # detrend_df_f(A, b, C, f) # Too slow!

        image, cor_frame = self.makeImage()
        dims = image.shape
        self._updateCoords(A,dims)

        # ids = self.client.random_ObjectID(4)
        # objs = [np.array(C), A.toarray(), image, np.array(cor_frame)]
        # for i,obj in enumerate(objs):
        #     try:
        #         self.client._put(obj, ids[i])
        #         print('put time: ', time.time()-t)
        #     except Exception as e:
        #         logger.error('_put exception {}'.format(e))    
        # print('total put time: ', time.time()-t)

        ids = []
        ids.append(self.client.put(np.array(C), str(self.frame_number)))
        ids.append(self.client.put(self.coords, 'coords'+str(self.frame_number)))
        ids.append(self.client.put(image, 'image'+str(self.frame_number)))
        ids.append(self.client.put(np.array(cor_frame), 'cor_frame'+str(self.frame_number)))

        self.q_out.put(ids)
        self.q_comm.put([self.frame_number])
    
    def _updateCoords(self, A, dims):
        '''See if we need to recalculate the coords
           Also see if we need to add components
        '''
        if self.coords is None: #initial calculation
            self.A = A
            self.coords = get_contours(A, dims)

        elif np.shape(A)[1] > np.shape(self.A)[1]: #Only recalc if we have new components
            self.A = A
            self.coords = get_contours(A, dims)

    def makeImage(self):
        '''Create image data for visualiation
            Using caiman code here
            #TODO: move to MeanAnalysis class ?? Check timing if we move it!
                Other idea -- easier way to compute this?
        '''
        mn = self.onAc.M - self.onAc.N
        image = None
        try:
            components = self.onAc.estimates.Ab[:,mn:].dot(self.onAc.estimates.C_on[mn:self.onAc.M,self.frame_number-1]).reshape(self.onAc.dims, order='F')
            background = self.onAc.estimates.Ab[:,:mn].dot(self.onAc.estimates.C_on[:mn,self.frame_number-1]).reshape(self.onAc.dims, order='F')
            image = ((components + background) - self.onAc.bnd_Y[0])/np.diff(self.onAc.bnd_Y)
            image = np.minimum((image*255.),255).astype('u1')
        except ValueError as ve:
            logger.info('ValueError: {0}'.format(ve))

        cor_frame = (self.frame - self.onAc.bnd_Y[0])/np.diff(self.onAc.bnd_Y)
        return image, cor_frame

    def _finalAnalysis(self, t):
        ''' Some kind of final steps for estimates
            t is the final frame count
            TODO: get params from elsewhere in case changed!
            Can't put these into onAc.estimates since it changes shape,
            and thus interferes with running the process again in a loop.
        '''
        #TODO change epochs?
        epochs = 1

        self.finalEstimates = self.onAc.estimates
        
        self.finalEstimates.A, self.finalEstimates.b = self.onAc.estimates.Ab[:, self.onAc.params.get('init', 'nb'):], self.onAc.estimates.Ab[:, :self.onAc.params.get('init', 'nb')].toarray()
        self.finalEstimates.C, self.finalEstimates.f = self.onAc.estimates.C_on[self.onAc.params.get('init', 'nb'):self.onAc.M, t - t //
                         epochs:t], self.onAc.estimates.C_on[:self.onAc.params.get('init', 'nb'), t - t // epochs:t]
        noisyC = self.onAc.estimates.noisyC[self.onAc.params.get('init', 'nb'):self.onAc.M, t - t // epochs:t]
        self.finalEstimates.YrA = noisyC - self.onAc.estimates.C
        self.finalEstimates.bl = [osi.b for osi in self.onAc.estimates.OASISinstances] if hasattr(self, 'OASISinstances') else [0] * self.onAc.estimates.C.shape[0]
        self.finalEstimates.C_on = self.onAc.estimates.C_on[:self.onAc.M]
        self.finalEstimates.noisyC = self.onAc.estimates.noisyC[:self.onAc.M]


    def _checkFrames(self):
        ''' Check to see if we have frames for processing
            TODO: rework logic since not accessing store directly here anymore
        '''
        try:
            res = self.q_in.get(timeout=0.005)
            return res
            #return self.client.get('frame')
        # except CannotGetObjectError:
        #     logger.error('No frames')
        #TODO: add'l error handling
        except Empty:
            #logger.info('no frames for processing')
            return None


    def _processFrame(self, frame, frame_number):
        ''' Do some basic processing on a single frame
            Raises NaNFrameException if a frame contains NaN
            Returns the normalized/etc modified frame
        '''
        t=time.time()
        if np.isnan(np.sum(frame)):
            raise NaNFrameException
        frame = frame.astype(np.float32) #or require float32 from image acquistion
        if self.onAc.params.get('online', 'ds_factor') > 1:
            frame = cv2.resize(frame, self.onAc.img_norm.shape[::-1])
            # TODO check for params, onAc componenets before calling, or except
        if self.onAc.params.get('online', 'normalize'):
            frame -= self.onAc.img_min
        if self.onAc.params.get('online', 'motion_correct'):
            try:
                templ = self.onAc.estimates.Ab.dot(
                self.onAc.estimates.C_on[:self.onAc.M, frame_number-1]).reshape(
                self.onAc.params.get('data', 'dims'), order='F')*self.onAc.img_norm
            except Exception as e:
                logger.error('Unknown exception {0}'.format(e))
                raise Exception
            
            if self.onAc.params.get('motion', 'pw_rigid'):
                frame_cor1, shift = motion_correct_iteration_fast(
                            frame, templ, self.max_shifts_online, self.max_shifts_online)
                frame_cor, shift = tile_and_correct(frame, templ, self.onAc.params.motion['strides'], self.onAc.params.motion['overlaps'], self.onAc.params.motion['max_shifts'], newoverlaps=None, newstrides=None, upsample_factor_grid=4, upsample_factor_fft=10, show_movie=False, max_deviation_rigid=self.onAc.params.motion['max_deviation_rigid'],add_to_movie=0, shifts_opencv=True, gSig_filt=None, use_cuda=False, border_nan='copy')[:2]
            else:
                frame_cor, shift = motion_correct_iteration_fast(frame, templ, self.max_shifts_online, self.max_shifts_online)
            self.onAc.estimates.shifts.append(shift)
        else:
            frame_cor = frame
        if self.onAc.params.get('online', 'normalize'):
            frame_cor = frame_cor/self.onAc.img_norm
        self.procFrame_time.append([time.time()-t])
        return frame_cor

    def _fitFrame(self, frame_number, frame):
        ''' Do the heavy lifting here. CNMF, etc
            Updates self.onAc.estimates
        '''
        try:
            self.onAc.fit_next(frame_number, frame)
        except Exception as e:
            print('Message: {0}'.format(e))
            raise Exception

    def _normAnalysis(self):
        ''' Modifies in place Ab!
        Do not use until the end, or
        TODO: just return nicely normalized Ab. Possibly combine with other
            visualization/estimation output
        '''
        self.onAc.estimates.Ab /= 1./self.onAc.img_norm.reshape(-1, order='F')[:,np.newaxis]
        self.onAc.estimates.Ab = scipy.sparse.csc_matrix(self.onAc.estimates.Ab)


class NaNFrameException(Exception):
    pass








