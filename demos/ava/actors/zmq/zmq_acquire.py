import numpy as np
import os
import pandas as pd
import time

from dvg_ringbuffer import RingBuffer

from actors.zmq.zmq_actor import ZMQActor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ZMQAcquirer(ZMQActor):
    """Actor for acquiring audio as ZMQ messages sent from LabVIEW-EvTAF18 sent over TCP to improv.
    """

    def __init__(self, *args, ip=None, port=None, msg_type="multipart", fs=None, win_dur=None, n_lag=0, max_len=2000, dtype="int16", time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.msg_type = msg_type
        self.ip = ip
        self.port = port

        self.time_opt = time_opt
        if self.time_opt:
            self.timing = timing
            self.timing_path = os.path.join(out_path, 'timing')
            logger.info(f"{self.name} timing directory: {self.timing_path}")
        
        self.n_per_seg = int(win_dur * fs)
        self.n_lag = n_lag # 10 ms
        logger.info(f"n_per_seg: {self.n_per_seg}")

        self.max_len = int(max_len)
        self.dtype = dtype
        if self.dtype == "int16":
            self.dtype == np.int16

        self.done = False

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """_summary_
        """

        logger.info(f"Running setup for {self.name}.")

        self.setRecvSocket(self.ip, self.port, timeout=0)

        if self.time_opt:
            os.makedirs(self.timing_path, exist_ok=True)

            logger.info(f"Initializing lists for {self.name} timing.")
            
            self.zmq_acq_timestamps = []
            self.recv_msg = []
            self.ext_buffer = []
            self.get_data = []
            self.put_seg_to_store = []
            self.put_out_time = []
            self.zmq_acq_total_times = []
            self.ns_per_msg = []
            self.ns_overlap = []
            self.ns_dropped = []

        self.dropped_msgs = []

        logger.info(f"Initializing ring buffer with max_len: {self.max_len} supporting dtype: {self.dtype}.")
        self.buffer = RingBuffer(capacity=self.max_len, dtype=self.dtype)

        self.msg_num = 0
        self.seg_num = 0

        logger.info(f"Completed setup for {self.name}.")
    
    def stop(self):
        
        logger.info(f"{self.name} stopping.")

        logger.info(f"Acquirer avg time per segment: {np.mean(self.zmq_acq_total_times)}")
        logger.info(f"Acquirer got through {self.msg_num} messages and {self.seg_num} segments.")

        # Close subscriber socket
        self.recv_socket.close()

        if self.time_opt:
            logger.info(f"Saving timing info for {self.name}.")
            keys = self.timing
            values = [self.zmq_acq_timestamps, self.recv_msg, self.ext_buffer, self.get_data, self.put_seg_to_store, self.put_out_time, self.zmq_acq_total_times, self.ns_per_msg, self.ns_overlap, self.ns_dropped]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.timing_path, 'acq_timing_' + str(self.seg_num) + '.csv'), index=False, header=True)
        
        logger.info(f"{self.name} stopped.")
        
        return 0
    
    def runStep(self):
        """_summary_
        """

        if self.done:
            pass

        t = time.perf_counter_ns()

        if self.time_opt:
            self.zmq_acq_timestamps.append(time.perf_counter_ns())

        try:

            t1 = time.perf_counter_ns()
            msg = self.recvMsg(self.msg_type)
            n = len(msg)
            # self.ns_per_msg.append(n)
            t2 = time.perf_counter_ns()

            self.buffer.extend(msg)
            t3 = time.perf_counter_ns()

            # fine to be overwritten — point of buffer
            # if len(self.buffer) + n > self.max_len:
            #     logger.info("Data will be overwritten: len(buffer) + len(msg) > max_len of buffer.")
            
            # will never happen...
            if n > self.max_len:
               logger.info("Data will be lost: len(msg) > max_len of buffer.")

            if n > self.n_per_seg:
                logger.info("Data will be lost: len(msg) > n_per_seg")
                self.ns_dropped.append(int(n - self.n_per_seg))

            if len(self.buffer) >= self.n_per_seg:

                if self.seg_num == 0:
                    self.ns_dropped.append(int(len(self.buffer) - self.n_per_seg))
                    self.ns_overlap.append(int(0))

                # if n < self.n_lag:
                #     msg = self.recvMsg(self.msg_type)
                #     # t2 = time.perf_counter_ns()

                #     self.buffer.extend(msg)
                #     n += len(msg)

                overlap = self.n_per_seg - n
                    
                    # if self.n_lag != 0:
                    #     while overlap > self.n_lag:
        
                if overlap < 0:
                    self.ns_dropped.append(int(-overlap))
                    self.ns_overlap.append(0)
                else:
                    self.ns_dropped.append(0)
                    self.ns_overlap.append(int(overlap))

                self.ns_per_msg.append(n)

                t4 = time.perf_counter_ns()
                self.data = self.buffer[-self.n_per_seg:]
                t5 = time.perf_counter_ns()

                data_obj_id = self.client.put(self.data, f"seg_num_{str(self.seg_num)}")
                t6 = time.perf_counter_ns()

                self.q_out.put(data_obj_id)
                t7 = time.perf_counter_ns()

                if self.time_opt:
                    self.get_data.append((t5 - t4) * 10**-3)
                    self.put_seg_to_store.append((t6 - t5) * 10**-3)
                    self.put_out_time.append((t7 - t6) * 10**-3)

                # logger.info(f"data size: {self.client.get_all()[data_obj_id]['data_size']}")
                
                # logger.info(f"data: {self.data} \
                #             \nlen(data): {len(self.data)}")

                self.seg_num += 1
                
            elif len(self.buffer) < self.n_per_seg:
                self.get_data.append(np.nan)
                self.put_seg_to_store.append(np.nan)
                self.put_out_time.append(np.nan)

            if self.time_opt:
                self.recv_msg.append((t2 - t1) * 10**-3)
                self.ext_buffer.append((t3- t2) * 10**-3)

            self.msg_num += 1

        except Exception as e:
            logger.error(f"Acquirer general exception: {e}")
        # except IndexError as e:
        #     pass

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
        self.zmq_acq_total_times.append((time.perf_counter_ns() - t) * 10**-3)