from collections import deque # for deque circular/ring buffer, remove rear element, add front element
import numpy as np
import os
import pandas as pd
import time
import traceback
import zmq

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class ZMQAcquirer(Actor):
class ZMQSubscribe(Actor):

    def __init__(self, *args, ip_address=None, port=None, fs=None, win_dur=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # single, global Context instance
        # for classes that depend on Contexts, use a default argument to enable programs with multiple Contexts, not require argument for simpler applications
        # called in a subprocess after forking, a new global instance is created instead of inheriting a Context that won’t work from the parent process
        # add self.ctx for global, different methods = PUB/SUB
        ctx = zmq.Context.instance()

        self.subscriber = ctx.socket(zmq.SUB)
        # error handling: EINVAL, socket type invalie; EFAULT, context invalid; EMFILE, limit num of open sockets; ETERM, context terminated
        self.subscriber.setsockopt(zmq.SUBSCRIBE,b"")
        # 
        # connect client node, socket,  with unkown or arbitrary network address(es) to endpoint with well-known network address
        # connect socket to peer address
        # endpoint = peer address:TCP port, source_endpoint:'endpoint'
        # IPv4/IPv6 assigned to interface OR DNS name:TCP port
        self.subscriber.connect(f"tcp://{ip_address}:{port}")

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
        
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

        self.data = deque([], maxlen=int(self.seg_dur))
        # self.data = []

        with RunManager(self.name, self.runZMQAcquirer, self.setup, self.q_sig, self.q_comm, self.stop) as rm:
            print(rm)
        
        print('Acquire broke, avg time per segment:', np.mean(self.zmq_acq_total_times))
        print('Acquire got through', self.seg_num, 'segments')

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
            try:
                msg = self.sock.recv_multipart()
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    print("No msg ready")
                    pass  # no message was ready
                else:
                    raise  # real error
            self.data.extend(np.int16(msg[:-1]))
            self.ns.append(int(msg[-1]))
            if np.sum(self.ns) >= int(self.seg_dur) * self.seg_num+1:
                t2 = time.time()
                self.get_data.append((t2 - t1)*1000.0)
                t3 = time.time()
                data_obj_id = self.client.put(np.array(self.data), 'seg_num_' + str(self.seg_num))
                seg_obj_id = self.client.put(self.seg_num, 'seg_num_' + str(self.seg_num))
                t4 = time.time()
                self.put_to_store.append((t4 - t3)*1000.0)
                t5 = time.time()
                self.q_out.put([data_obj_id, seg_obj_id])
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

                # def subscribe(self):
        #     self.subscriber = self.ctx.socket(zmq.SUB)
        #     self.subscriber.setsockopt(zmq.SUBSCRIBE,b"")
        #   self.subscriber.connect(f"tcp://{ip_address}:{port}")

    