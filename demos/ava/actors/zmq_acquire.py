import zmq
import numpy as np
import pandas as pd
import os
import time

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):
    '''
    '''
    def __init__(self, *args, ip_address=None, port=None, fs=None, win_dur=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # self.ip_address = ip_address
        # self.port = port

        ctx = zmq.Context()
        # self.ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE,"")
        # Make sure ip_address and port are str
        self.sock.connect("tcp://" + str(ip_address) + ':' + str(port))
        # self.sock.connect("tcp://" + str(self.ip_address) + str(self.port))

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
        
        self.seg_dur = (fs*win_dur)/1000.0


    def setup(self):
        '''
        '''
        if self.time_opt is True:
            os.makedirs(self.out_path, exist_ok=True)

    def stop(self):
        '''
        '''
        self.sock.close()

        del self.data_buffer

        print(self.time_opt)
        values = [self.in_timestamps, self.zmq_acq_total_times, self.zmq_timestamps, self.get_data, self.put_to_store, self.put_out_time]
        print(values)

        if self.time_opt is True:
            keys = self.timing
            values = [self.in_timestamps, self.zmq_acq_total_times, self.zmq_timestamps, self.get_data, self.put_to_store, self.put_out_time]
            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'zmq_acq_timing.csv'), index=False, header=True)

        return 0
            
    def run(self):
        ''' Triggered at Run
        '''
        self.in_timestamps = []
        self.zmq_acq_total_times = []
        self.zmq_timestamps = []
        self.get_data = []
        self.put_to_store = []
        self.put_out_time = []

        self.seg_num = 0

        self.data_buffer = []

        with RunManager(self.name, self.runZMQAcquirer, self.setup, self.q_sig, self.q_comm, self.stop) as rm:
            print(rm)
        
        del self.data_buffer

        print('Acquire broke, avg time per segment:', np.mean(self.zmq_acq_total_times))
        print('Acquire got through', self.seg_num, ' segments')


    def runZMQAcquirer(self):
        '''
        '''
        if self.time_opt is True:
            self.zmq_timestamps.append(time.time())

        if self.done:
            pass

        t = time.time()

        try:   
            t1 = time.time()
            msg = self.sock.recv_multipart()
            array = np.asarray(msg)
            curr_data = np.frombuffer(array, dtype=np.int16)
            self.in_timestamps.append(curr_data[0])
            curr_data = curr_data[1:]
            self.data_buffer = np.concatenate((self.data_buffer, curr_data), axis=0)
            onset = int(self.seg_num*self.seg_dur)
            offset = int((self.seg_num+1)*self.seg_dur)
            print(self.data_buffer.size, offset)
            if self.data_buffer.size >= offset:
                data = self.data_buffer[onset:offset]
                t2 = time.time()
                self.get_data.append((t2 - t1)*1000.0)
                print(self.get_data)
                t3 = time.time()
                data_obj_id = self.client.put(data, 'seg_num_' + str(self.seg_num))
                seg_obj_id = self.client.put(self.seg_num, 'seg_num_' + str(self.seg_num))
                t4 = time.time()
                self.put_to_store.append((t4 - t3)*1000.0)
                print(self.put_to_store)
                t5 = time.time()
                self.q_out.put([data_obj_id, seg_obj_id])
                self.put_out_time.append((time.time() - t5)*1000.0)
                print(self.put_out_time)
                self.data_buffer[onset:offset] = None
                self.seg_num += 1
                
                self.zmq_acq_total_times.append((time.time() - t)*1000.0)
                print(self.zmq_acq_total_times)

        except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
        except IndexError as e:
            pass