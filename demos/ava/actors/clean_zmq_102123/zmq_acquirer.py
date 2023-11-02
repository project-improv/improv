import os
import time

from dvg_ringbuffer import RingBuffer
import numpy as np
import pandas as pd
import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, ETERM, EAGAIN, RCVBUF, RCVHWM

# from actors.zmq.zmq_actor import ZMQActor

from improv.actor import Actor

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ZMQAcquirer(Actor):
    """Actor for acquiring audio as ZMQ messages sent from LabVIEW-EvTAF18 over TCP to improv.
    """

    def __init__(self, *args, ip=None, port=None, msg_type="multipart", fs=None, win_dur=None, lag_dur=0, max_len=5000, dtype="int16", time_opt=True, timing=None, out_path=None, **kwargs):
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
        self.n_lag = int(lag_dur * fs) # arbitrary lag duration in ms
        logger.info(f"n_per_seg: {self.n_per_seg}")
        logger.info(f"n_lag: {self.n_lag}")

        self.max_len = int(max_len)
        self.dtype = dtype
        if self.dtype == "int16":
            self.dtype == np.int16

        self.done = False

        self.context = zmq.Context.instance()

        self.recv_address = f"tcp://{ip}:{port}"

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """Setup for AVA ZMQAudioAcquirer.
        """
        logger.info(f"Running setup for {self.name}.")

        self.recv_socket = self.context.socket(SUB)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        # self.recv_socket.setsockopt(RCVBUF, 212992)
        # self.recv_socket.set_hwm(1)
        self.recv_socket.connect(self.recv_address)
        logger.info(f"Subscriber socket set: {self.recv_address}")

        self.zmq_acq_total_times = []

        if self.time_opt:
            os.makedirs(self.timing_path, exist_ok=True)

            logger.info(f"Initializing lists for {self.name} timing.")
            
            self.zmq_acq_timestamps = []
            self.t_msgs = []
            self.t_recvs = []
            self.t_start_recv = []
            self.recv_seg = []
            self.ns_per_msgs = []
            self.ns_per_seg = []
            self.seg_nums = []

        self.dropped_msgs = []

        logger.info(f"Initializing ring buffer with max_len: {self.max_len} supporting dtype: {self.dtype}.")
        self.buffer = RingBuffer(capacity=self.max_len, dtype=self.dtype)

        self.msg_num = 0
        self.seg_num = 0

        logger.info(f"Completed setup for {self.name}.")
    
    def stop(self):
        """Stop procedure — close the SUB socket, save out timing information.
        """
        logger.info(f"{self.name} stopping.")

        logger.info(f"Acquirer avg time per segment: {np.mean(self.zmq_acq_total_times)}")
        logger.info(f"Acquirer got through {self.msg_num} messages and {self.seg_num} segments.")

        # Close subscriber socket
        self.recv_socket.close()

        if self.time_opt:
            logger.info(f"Saving timing info for {self.name}.")
            keys = self.timing
            values = [self.zmq_acq_timestamps, self.t_msgs, self.t_recvs, self.t_start_recv, self.recv_seg, self.ns_per_msgs, self.ns_per_seg, self.zmq_acq_total_times, self.seg_nums]
            
            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.timing_path, 'acq_timing_' + str(self.seg_num) + '.csv'), index=False, header=True)
            
        logger.info(f"{self.name} stopped.")
        
        return 0
    
    def runStep(self):
        """Run step — receive audio and associated DAQ timestamps as multipart messages via ZMQ, add messages to buffer, send out audio upon receiving at least 10 ms of data.
        """
        if self.done:
            pass

        t = time.perf_counter_ns()

        if self.time_opt:
            self.zmq_acq_timestamps.append(t)
            
        try:

            n = 0  # needs to be here, reset every run
            t_msgs = []
            t_recvs = []
            ns_per_msg = []
            t_start_recv = time.perf_counter_ns()
            while (self.seg_num == 0 and len(self.buffer) < self.n_per_seg) or (self.seg_num > 0 and n < self.n_lag):
                msg, t_msg, t_recv = self.recvMsg(self.msg_type) # t_msg DAQ timestamp, t_recv receive timestamp
                t_msgs.append(int(t_msg))
                t_recvs.append(int(t_recv))
                n += len(msg)
                ns_per_msg.append(len(msg))
                self.buffer.extend(msg)

            self.data = self.buffer[-self.n_per_seg:]

            data_obj_id = self.client.put(self.data, f"seg_num_{str(self.seg_num)}")
            timestamp_obj_id = self.client.put(int(t_msg), f"seg_num_{str(self.seg_num)}")

            self.q_out.put([data_obj_id, timestamp_obj_id])

            if self.time_opt:
                self.seg_nums.append(self.seg_num)
                self.t_start_recv.append(t_start_recv)
                self.recv_seg.append(t_recv)
                self.ns_per_msgs.append(ns_per_msg)
                self.ns_per_seg.append(n)
                self.t_msgs.append(t_msgs)
                self.t_recvs.append(t_recvs)

        except Exception as e:
            logger.error(f"Acquirer general exception: {e}")

        self.seg_num += 1
        self.zmq_acq_total_times.append((time.perf_counter_ns() - t) * 10**-6)

        self.data = None
        self.q_comm.put(None)
        self.done = True  # stay awake in case we get a shutdown signal

    def recvMsg(self, msg_type="multipart"):
        """Receive audio and associated DAQ timestamps as multipart messages via ZMQ.

        Args:
            msg_type (str, optional): _description_. Defaults to "multipart".

        Returns:
            msg: _description_
            t_msg: _description_
            t_recv: _description_
        """
        try:
            if msg_type == "multipart":
                msg = self.recv_socket.recv_multipart()
                t_recv = time.perf_counter_ns()
                t_msg = msg[-1]
                msg = msg[0:-1]
            else: 
                msg = self.recv_socket.recv()
        except ZMQError as e:
            logger.info(f"ZMQ error: {e}")
            if e.errno == ETERM:
                pass  # Interrupted - or break if in loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise  # real error
                # traceback.print_exc()
        
        return msg, t_msg, t_recv
