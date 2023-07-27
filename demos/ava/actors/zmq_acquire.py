import numpy as np
import os
import pandas as pd
import time

from improv.demos.ava.actors.ring_buffer import RingBuffer
from improv.demos.ava.actors.zmq.zmq_actor import ZMQActor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# redundant name?
class ZMQAcquirer(ZMQActor):
# class Acquirer(ZMQActor):
    """
    TODO CONSIDER POLLING
    https://github.com/zeromq/pyzmq/blob/main/examples/monitoring/zmq_monitor_class.py
    """
    # TODO: WHAT IS WITH THESE INIT VALUES?
    # NOTE: cannot import socket_type as arg!
    def __init__(self, *args, socket_type=2, ip=None, port=None, msg_type="multipart", fs=None, win_dur=None, n_overlap=0, max_len=2000, dtype="int16", time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # single, global Context instance
        # for classes that depend on Contexts, use a default argument to enable programs with multiple Contexts, not require argument for simpler applications
        # called in a subprocess after forking, a new global instance is created instead of inheriting a Context that won’t work from the parent process
        # add self.ctx for global, different methods = PUB/SUB
        # 

        self.socket = self.setSocket(socket_type)
        self.connectSocket(self.socket, ip, port)
        
        self.msg_type = msg_type

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
            
            os.makedirs(self.out_path, exist_ok=True)
            logger.info(f"Timing directory for {self.name}: {self.out_path}")
        
        # n per seg, samples for given window duration, for example, 20 ms of data at 32000 fs = 20 ms of data at 320 fms
        # or 0.02 s * 32000 fs
        self.n_per_seg = int(win_dur * fs) # * 10**-3 if entered in ms
        logger.info(f"n_per_seg: {self.n_per_seg}")
        self.n_overlap = n_overlap # if n points, n overlap
        # self.n_overlap = int(np.floor(n_overlap * fs))  # if ms, time = n_overlap
        logger.info(f"n_overlap: {self.n_overlap}")
        self.max_len = int(max_len)
        self.dtype = dtype
        # TODO: options for more dtypes for RingBuffer
        if self.dtype == "int16":
            self.dtype == np.int16

        self.done = False

    def __str__(self):
        # unnecessary?
        return f"Name: {self.name}, Data: {self.data}"


    def setup(self):
        """
        TODO HERE SYNC SUB w/PUB w/REQ/REP —> DO NOT WANT THAT!
        https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
        """

        logger.info(f"Running setup for {self.name}.")

        # initialize timing lists
        if self.time_opt is True:
            logger.info(f"Initializing lists for {self.name} timing.")
            
            self.zmq_acq_timestamps = []
            self.recv_msg = []
            self.get_data = []
            self.pop_data = []
            self.put_seg_to_store = []
            self.put_out_time = []
            self.zmq_acq_total_times = []
            self.ns_per_msg = []

        self.dropped_msg = []
        self.ns_per_msg = []

        logger.info(f"Initializing ring buffer for with max_len: {self.max_len} supporting dtype: {self.dtype}.")
        self.buffer = RingBuffer(capacity=self.max_len, dtype=self.dtype)

        # in __init__?
        self.msg_num = 0
        self.seg_num = 0

        logger.info(f"Completed setup for {self.name}.")

    def stop(self):
        """
        Meh... https://stackoverflow.com/questions/9019873/should-i-close-zeromq-socket-explicitly-in-python
        """
        
        logger.info(f"{self.name} stopping.")
        # Close subscriber socket
        self.closeSocket(self.socket)
        # Terminate context = ONLY terminate if/when BOTH SUB and PUB sockets are closed
        # logger.info("Terminating context.")
        # self.context.term()

        if self.time_opt is True:
            logger.info(f"Saving out timing info for {self.name}")
            keys = self.timing
            values = [self.zmq_acq_timestamps, self.recv_msg, self.get_data, self.pop_data, self.put_seg_to_store, self.put_out_time, self.zmq_acq_total_times, self.ns_per_msg]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'zmq_acq_timing.csv'), index=False, header=True)
        
        logger.info(f"{self.name} stopped")

        return 0
    
    def runStep(self):
        """
        """

        if self.done:
            pass

        t = time.perf_counter_ns()

        if self.time_opt is True: # cleaner w/o explicit is True?
            self.zmq_acq_timestamps.append(time.perf_counter_ns())

        try:   
            # logger.info(f"Receiving msg {self.msg_num}")

            t1 = time.perf_counter_ns()
            msg = self.recvMsg(self.socket, self.msg_type)
            t2 = time.perf_counter_ns()
            # logger.info(f"Msg received: {msg}")

            # EXTEND, REMOVE N SAMPLES AT END OF MSG W/ OR W/O OVERLAP, SLIDING WINDOW
            self.buffer.extend(msg)
            t3 = time.perf_counter_ns()

            # ONLY FOR TIMING
            self.ns_per_msg.append(len(msg))

            # logger.info(f"data: {self.data}")
            # logger.info(f"ns: {self.ns}")

            # FOR IF TAKING EXACT SEG LEN
            # if np.sum(self.ns_per_msg) >= self.n_per_seg * (self.seg_num + 1):
            # DATA MUST BE MIN LEN — take fixed, keep indexing, popping
            # FOR THE FIRST SEG, CONTINUE after arbitrary n_overlap
            logger.info(f"buffer: {self.buffer}, \n, len(buffer): {len(self.buffer)}")
            if len(self.buffer) >= self.n_per_seg:
                logger.info(f"len(buffer): {self.n_per_seg}")
                t4 = time.perf_counter_ns()
                self.data = self.buffer.popleft(n=self.n_per_seg, n_overlap=self.n_overlap)
                t5 = time.perf_counter_ns()
                data_obj_id = self.client.put(self.data, f"seg_num_{str(self.seg_num)}")
                t6 = time.perf_counter_ns()
                self.q_out.put(data_obj_id)
                self.put_out_time.append((time.perf_counter_ns() - t6)* 10**-3)

                # logger.info(f"data size: {self.client.get_all()[data_obj_id]['data_size']}")

                self.pop_data.append((t5 - t4)*1000.0)
                self.put_seg_to_store.append((t6 - t5) * 10**-3)
                
                self.data = self.data[self.skip_len:]
                logger.info(f"data: {self.data}, \n, len(data): {len(self.data)}")
                self.seg_num += 1

            self.recv_msg.append((t2 - t1) * 10**-3)
            self.get_data.append((t3- t2) * 10**-3)

            self.msg_num += 1

        except Exception as e:
                logger.error(f"Acquirer general exception: {e}")
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
        self.zmq_acq_total_times.append((time.perf_counter_ns() - t) * 10**-3)

        logger.info(f"{self.name} avg time per run: {np.mean(self.zmq_acq_total_times)} µs")

        # logger.info(f"Acquire broke, avg time per segment: {np.mean(self.zmq_acq_total_times)} ms")
        # logger.info(f"Acquire got through {self.seg_num} segments")