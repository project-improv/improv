import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Classes: File Acquirer, Stim, Behavior, Tiff

class FileAcquirer(Actor):
    # TODO: Make file type-/extension-agnostic?
    '''Class to import data from files and output
       frames in a buffer, or discrete.
    '''
    def __init__(self, *args, filename=None, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.flag = False
        self.filename = filename
        self.framerate = 1/framerate

    def setup(self):
        '''Get file names from config or user input
        Also get specified framerate, or default is 10 Hz
        Open file stream
        # TODO: implement more than .h5 files
        '''
        print('Looking for ', self.filename)
        if os.path.exists(self.filename):
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == '.h5' or ext == '.hdf5':
                with h5py.File(self.filename, 'r') as file:
                    keys = list(file.keys())
                    self.data = file[keys[0]][()]
                    print('Data length is ', len(self.data))

        else: raise FileNotFoundError

    def run(self):
        ''' Run indefinitely. Calls runAcquirer after checking for signals
        '''
        self.total_times = []
        self.timestamp = []

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        print('Done running Acquire, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.frame_num, ' frames')
        if not os._exists('output'):
            try:
                os.makedirs('output')
            except:
                pass
        if not os._exists('output/timing'):
            try:
                os.makedirs('output/timing')
            except:
                pass
        np.savetxt('output/timing/acquire_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/acquire_timestamp.txt', np.array(self.timestamp))

    def runAcquirer(self):
        '''While frames exist in location specified during setup,
           grab frame, save, put in store
        '''
        t = time.time()

        if self.done:
            pass
        elif(self.frame_num < len(self.data)):
            frame = self.getFrame(self.frame_num)
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            t = time.time()
            id = self.client.put(frame, 'acq_raw' + str(self.frame_num))
            t1 = time.time()
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.put([[id, str(self.frame_num)]], save=[True])
                self.frame_num += 1
                 #also log to disk #TODO: spawn separate process here?
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))

            time.sleep(self.framerate) # pretend framerate
            self.total_times.append(time.time()-t)

        else: # simulating a done signal from the source (eg, camera)
            logger.error('Done with all available frames: {0}'.format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True # stay awake in case we get e.g. a shutdown signal
            #if self.saving:
            #    self.f.close()

    def getFrame(self, num):
        ''' Here just return frame from loaded data
        '''
        return self.data[num,:,:]

    def saveFrame(self, frame):
        ''' Save each frame via h5 dset
        '''
        self.dset[self.frame_num-1] = frame
        self.f.flush()

class FolderAcquirer(Actor):
    # TODO: Make file type-/extension-agnostic?
    ''' Current behavior is looping over all files in a folder.
        Class to read TIFF/JPG/PNG files in a specified {path} from disk.
        Designed for scenarios when new TIFF files are created during the run.
        Reads only new TIFF files (after Run started) and put on to the Plasma store.
        If there're multiple files, files are loaded by name.
    '''

    def __init__(self, *args, folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.done = False
        self.path = Path(folder)

        # self.frame_num = 0
        self.sample_num = 0
        self.files = []

        if not self.path.exists() or not self.path.is_dir():
            raise AttributeError('Folder {} does not exist.'.format(self.path))

    def setup(self):
        pass
        
    def saveImgs(self, exts):
        '''
        Arg: exts = list of possible file extensions
        '''
        self.imgs = []
        files = {f for f in self.path.iterdir() if f.suffix in exts}
        files = sorted(list(files))
        for file in files:
            img = self.get_sample(file)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs)
        f = h5py.File('output/sample.h5', 'w', libver='latest')
        f.create_dataset("default", data=self.imgs)
        f.close()

    def run(self, exts):
        ''' Triggered at Run
            Get list of files in the folder and use that as the baseline.
            Arg: exts = list of possible extensions
        '''
        self.total_times = []
        self.timestamp = []
        
        self.files = [f for f in self.path.iterdir() if f.suffix in exts]

        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

        if '.tif' or '.tiff' in exts:
            print('Acquire broke, avg time per frame: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' frames')
        if '.jpg' or '.png' in exts:
            print('Acquire broke, avg time per image: ', np.mean(self.total_times))
            print('Acquire got through ', self.sample_num, ' images')

    def runAcquirer(self):
        ''' Main loop. If there're new files, read and put into store.
        '''
        if self.done:
            pass
        
        else:
            t = time.time()
            try:
                obj_id = self.client.put(self.get_sample(self.files[self.sample_num]), 'acq_raw' + str(self.sample_num))
                self.timestamp.append([time.time(), self.sample_num])
                self.put([[obj_id, str(self.sample_num)]], save=[True])
                self.sample_num += 1
            except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
            except IndexError as e:
                pass

            self.total_times.append(time.time() - t)
        
        # From bubblewrap demo and FileAcquirer
        logger.error('Done with all available data: {0}'.format(self.sample_num))
        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    def get_sample(self, file: Path):
        try:
            img = imread(file.as_posix())
        except ValueError as e:
            img = imread(file.as_posix())
            logger.error('File ' + file.as_posix() + ' had value error {}'.format(e))
        return img # [0,0,0, :, :,0]  # Extract first channel in this image set, or tensor/3D array for .jpg/.png

class StimAcquirer(Actor):
    ''' Class to load visual stimuli data from file
        and stream into the pipeline
    '''
    def __init__(self, *args, param_file=None, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_file = param_file
        self.filename = filename

    def setup(self):
        self.n= 0
        self.sID = 0
        if os.path.exists(self.filename):
            print('Looking for ', self.filename)
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == ".txt":
                # self.stim = np.loadtxt(self.filename)
                self.stim=[]
                f = np.loadtxt(self.filename)
                for _, frame in enumerate(f):
                    stiminfo = frame[0:2]
                    self.stim.append(stiminfo)
            else:
                logger.error('Cannot load file, possible bad extension')
                raise Exception

        else: raise FileNotFoundError

    def run(self):
        ''' Run continuously, waiting for input
        '''
        with RunManager(self.name, self.getInput, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

    def getInput(self):
        ''' Check for input from behavioral control
        '''
        if self.n<len(self.stim):
            # s = self.stim[self.sID]
            # self.sID+=1
            self.q_out.put({self.n:self.stim[self.n]})
        time.sleep(0.5)   # simulate a particular stimulus rate
        self.n+=1

class BehaviorAcquirer(Actor):
    ''' Actor that acquires information of behavioral stimulus
        during the experiment

        Current assumption is that stimulus is off or on, on has many types,
        and any change in stimulus is _un_associated with a frame number.
        TODO: needs to be associated with time, then frame number
    '''

    def __init__(self, *args, param_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_file = param_file

    def setup(self):
        ''' Pre-define set of input stimuli
        '''
        self.n = 0 #our fake frame number here
        #TODO: Consider global frame_number in store...or signal from Nexus

        #TODO: Convert this to Tweak and load from there
        if self.param_file is not None:
            try:
                params_dict = None #self._load_params_from_file(param_file)
            except Exception as e:
                logger.exception('File cannot be loaded. {0}'.format(e))
        else:
            self.behaviors = [0, 1, 2, 3, 4, 5, 6, 7] #8 sets of input stimuli

    def run(self):
        ''' Run continuously, waiting for input
        '''
        with RunManager(self.name, self.getInput, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

    def getInput(self):
        ''' Check for input from behavioral control
        '''
        # Faking it for now.
        if self.n % 50 == 0:
            self.curr_stim = random.choice(self.behaviors)
            self.onoff = random.choice([0,10])
            self.q_out.put({self.n:[self.curr_stim, self.onoff]})
            logger.info('Changed stimulus! {}'.format(self.curr_stim))
        time.sleep(1) #0.068)
        self.n += 1

class FileStim(Actor):
    ''' Actor that acquires information of behavioral stimulus
        during the experiment from a file
    '''

    def __init__(self, *args, File= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.file= File

    def setup(self):
        ''' Pre-define set of input stimuli
        '''
        self.n = 0 #our fake frame number here
        #TODO: Consider global frame_number in store...or signal from Nexus

        self.data= np.loadtxt(self.file)

    def run(self):
        ''' Run continuously, waiting for input
        '''
        with RunManager(self.name, self.getInput, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)

    def getInput(self):
        ''' Check for input from behavioral control
        '''
        # Faking it for now.
        if self.n % 50 == 0 and self.n< self.data.shape[1]*50:
            self.curr_stim = self.data[0][int(self.n/50)]
            self.onoff = self.data[1][int(self.n/50)]
            self.q_out.put({self.n:[self.curr_stim, self.onoff]})
            logger.info('Changed stimulus! {}'.format(self.curr_stim))
        time.sleep(0.068)
        self.n += 1

class TiffAcquirer(Actor):
    ''' Loops through a TIF file.
    '''

    def __init__(self, *args, filename=None, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

        if not os.path.exists(filename):
            raise ValueError('TIFF file {} does not exist.'.format(filename))
        self.imgs = None #np.array(0)

        self.n_frame = 0
        self.fps = framerate

        self.t_per_frame = list()

    def setup(self):
        self.imgs = imread(self.filename)
        print(self.imgs.shape)

    def run(self):
        with RunManager(self.name, self.run_acquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

    def run_acquirer(self):
        t0 = time.time()
        id_store = self.client.put(self.imgs[self.n_frame], 'acq_raw' + str(self.n_frame))
        self.q_out.put([[id_store, str(self.n_frame)]])
        self.n_frame += 1

        time.sleep(1 / self.fps)

        self.t_per_frame.append(time.time() - t0)