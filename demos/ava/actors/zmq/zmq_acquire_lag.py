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

    def __init__(self, *args, ip=None, port=None, msg_type="multipart", fs=None, win_dur=None, lag_dur=0, max_len=2000, dtype="int16", time_opt=True, timing=None, out_path=None, **kwargs):
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
        self.n_lag = int(lag_dur * fs) # 10 ms
        logger.info(f"n_per_seg: {self.n_per_seg}")
        logger.info(f"n_lag: {self.n_lag}")

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
            self.recv_msgs = []
            self.recv_seg = []
            self.get_data = []
            self.put_seg_to_store = []
            self.put_out_time = []
            self.zmq_acq_total_times = []
            self.ns_per_msgs = []
            self.ns_per_seg = []
            self.ns_overlap = []
            self.ns_dropped = []
            self.seg_nums = []
            self.timestamps = []

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
            values = [self.zmq_acq_timestamps, self.recv_msgs, self.recv_seg, self.get_data, self.put_seg_to_store, self.put_out_time, self.zmq_acq_total_times, self.ns_per_seg, self.ns_overlap, self.ns_dropped, self.seg_nums, self.timestamps]

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
            self.zmq_acq_timestamps.append(t)
            
            try:
                self.seg_nums.append(self.seg_num)

                n = 0  # needs to be here, reset every run
                ns_per_msg = []
                end_recv = 0
                t1 = time.perf_counter_ns()
                while (self.seg_num == 0 and len(self.buffer) < self.n_per_seg) or (self.seg_num > 0 and n < self.n_lag):
                    start_recv = time.perf_counter_ns()
                    msg = self.recvMsg(self.msg_type)
                    end_recv += (time.perf_counter_ns() - start_recv) * 10**-3
                    n += len(msg)
                    self.buffer.extend(msg)
                    ns_per_msg.append(len(msg))
                t2 = time.perf_counter_ns()
                
                # The following will never happen — exclude or important checks?
                # if n > self.max_len:
                #     logger.info("Data will be lost: len(msg) > max_len of buffer.")

                # if n > self.n_per_seg:
                #     logger.info("Data will be lost: len(msg) > n_per_seg")
                #     self.ns_dropped.append(int(n - self.n_per_seg))
                
                if self.seg_num == 0:
                    self.ns_dropped.append(int(len(self.buffer) - self.n_per_seg))
                    self.ns_overlap.append(int(0))

                # Should always be > n_per_lag
                overlap = self.n_per_seg - n

                if overlap < 0:
                    self.ns_dropped.append(int(-overlap))
                    self.ns_overlap.append(0)
                else:
                    self.ns_dropped.append(0)
                    self.ns_overlap.append(int(overlap))

                t3 = time.perf_counter_ns()
                self.data = self.buffer[-self.n_per_seg:]
                t4 = time.perf_counter_ns()

                data_obj_id = self.client.put(self.data, f"seg_num_{str(self.seg_num)}")
                t5 = time.perf_counter_ns()

                self.q_out.put(data_obj_id)
                t6 = time.perf_counter_ns()

                self.seg_num += 1

                if self.time_opt:
                    timestamps = [t1, t2, t3, t4, t5, t6]
                    self.timestamps.append(timestamps)
                    self.ns_per_msgs.append(ns_per_msg)
                    self.ns_per_seg.append(n)
                    self.recv_msgs.append(end_recv)
                    self.recv_seg.append((t2 - t1) * 10**-3)
                    self.get_data.append((t4 - t3) * 10**-3)
                    self.put_seg_to_store.append((t5 - t4) * 10**-3)
                    self.put_out_time.append((t6 - t5) * 10**-3)

                # logger.info(f"data size: {self.client.get_all()[data_obj_id]['data_size']}")
                
                # logger.info(f"data: {self.data} \
                #             \nlen(data): {len(self.data)}")

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
