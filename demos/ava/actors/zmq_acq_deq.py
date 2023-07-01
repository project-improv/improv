from collections import deque
import numpy as np
import os
import pandas as pd
import time
import traceback
import zmq

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):

    def __init__(self, *args, sub_ip=None, sub_port=None, seg=False, fs=None, win_dur=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        context = zmq.Context.instance()
        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.SUBSCRIBE,b"")
        self.subscriber.set_hwm(1000)
        self.subscriber.connect(f"tcp://{sub_ip}:{sub_port}")

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
        
        if seg is True:
            # win_dur in ms, for example, 20 ms
            self.seg_dur = (fs*win_dur)/1000.0

    def setup(self):
        '''
        '''
        if self.time_opt is True:
            os.makedirs(self.out_path, exist_ok=True)

    def stop(self):
        '''
        '''

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

        self.dropped_msg = []
        self.ns = []

        self.seg_num = 0
        self.i = 0

        self.data_obj_id = []
        self.seg_obj_id = []

        # self.data = deque([], maxlen=int(self.seg_dur))
        # Feeding 107 segments of max len 640/q_in
        self.data = deque([], maxlen=100)
        # self.data = deque([])
        # self.data = []

        with RunManager(self.name, self.runZMQAcquirer, self.setup, self.q_sig, self.q_comm, self.stop) as rm:
            print(rm)
        
        print('Acquire broke, avg time per segment:', np.mean(self.zmq_acq_total_times))
        print('Acquire got through', self.seg_num, 'segments')

    def runSetup(self):

        self.in_timestamps = []
        self.zmq_acq_total_times = []
        self.zmq_timestamps = []
        self.get_data = []
        self.put_to_store = []
        self.put_out_time = []

        self.dropped_msg = []
        self.ns = deque([], maxlen=10000)

        self.seg_num = 0
        self.i = 0

        self.data_obj_id = []
        self.seg_obj_id = []

        # self.data = deque([], maxlen=int(self.seg_dur))
        # Feeding 107 segments of max len 640/q_in
        self.data = deque([], maxlen=100)

    # def runStep(self):
    def runZMQAcquirer(self):
        '''
        '''
        # if self.time_opt is True:
            # self.zmq_timestamps.append(time.time())

        if self.done:
            pass

        t = time.time()
        try:   
            t1 = time.time()
            # try: if error worked
            try:
                msg = self.subscriber.recv_multipart()
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    print("error?")
                    # LOGGER.debug(f"e?"")
                    pass
            # except zmq.ZMQError as e:
            #     if e.errno == zmq.EAGAIN:
            #         print("No msg ready")
            #         pass  # no message was ready
            #     else:
            #         raise  # real error
            print(msg)
            # self.data.extend(np.int16(msg[:-1]))
            # print(f"data acq: {self.data} \n \n ")
            # self.ns.append(int(msg[-1]))
            # print(f"ns: {self.ns} \n \n", \
            #       f"N: {np.sum(self.ns)}", \
            #         f"len(ns): {len(self.ns)}")          
            # if np.sum(self.ns) >= 32000:
                # one = time.time()
                # print((time.time() - t - 1) * 1000)
                # time.sleep(2)
                # return one
            # else: return 0
            # time.sleep(1)
            # if np.sum(self.ns) >= int(self.seg_dur) * self.seg_num+1:
            print(f"seg_n: {self.seg_num}")
            t2 = time.time()
            self.get_data.append((t2 - t1)*1000.0)
            t3 = time.time()
            # self.data_obj_id.append(self.client.put(list(self.data), 'seg_num_' + str(self.seg_num)))
            # self.seg_obj_id.append(self.client.put(self.seg_num, 'seg_num_' + str(self.seg_num)))
            t4 = time.time()
            # self.put_to_store.append((t4 - t3)*1000.0)
            t5 = time.time()
            # print(f"out: {self.data_obj_id[-1]}, {self.seg_obj_id[-1]}")
            # self.q_out.put([self.data_obj_id[-1], self.seg_obj_id[-1]])
            # self.client.delete([self.data_obj_id[:-2], self.seg_obj_id[:-2]])
            # print(self.client.get_all()[data_obj_id, seg_obj_id]['data_size'])
            self.put_out_time.append((time.time() - t5)*1000.0)

            self.seg_num += 1
            
            self.zmq_acq_total_times.append((time.time() - t)*1000.0)

        except Exception as e:
                logger.error('Acquirer general exception: {}'.format(e))
        except IndexError as e:
            pass

        # # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
        # except ObjectNotFoundError:
        #     logger.error('Acquirer: Message {} unavailable from store, dropping'.format(self.seg_num))
        #     self.dropped_msg.append(self.seg_num)
        #     # self.q_out.put([1])
        # except KeyError as e:
        #     logger.error('Processor: Key error... {0}'.format(e))
        #     # Proceed at all costs
        #     self.dropped_wav.append(self.seg_num)
        # except Exception as e:
        #     logger.error('Processor error: {}: {} during segment number {}'.format(type(e).__name__,
        #                                                                                 e, self.seg_num))
        #     print(traceback.format_exc())
        #     self.dropped_wav.append(self.seg_num)
        self.zmq_acq_total_times.append((time.time() - t)*1000.0)
        # else:
        #     pass



if __name__ == "__main__":

    sub_ip = "10.122.168.184"
    sub_port = "5555"

    time_opt = False

    zmq_acq = ZMQAcquirer(name="ZMQAcq", sub_ip=sub_ip, sub_port=sub_port, time_opt=time_opt)

    zmq_acq.runSetup()
    
    t = time.time()
    while 1:
        zmq_acq.runZMQAcquirer()
        # one = zmq_acq.runZMQAcquirer()
        # print(one - t - 1)
