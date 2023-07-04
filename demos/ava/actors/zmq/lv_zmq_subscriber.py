from collections import deque # for deque circular/ring buffer, remove rear element, add front element
import numpy as np
import os
import pandas as pd
import time
# import traceback

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, ZMQError, ETERM, EAGAIN
from zmq.log.handlers import PUBHandler

from improv.actor import Actor, RunManager

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):
    """
    TODO CONSIDER POLLING
    https://github.com/zeromq/pyzmq/blob/main/examples/monitoring/zmq_monitor_class.py
    """

    def __init__(self, *args, ip=None, port=None, multipart=True, fs=None, win_dur=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # single, global Context instance
        # for classes that depend on Contexts, use a default argument to enable programs with multiple Contexts, not require argument for simpler applications
        # called in a subprocess after forking, a new global instance is created instead of inheriting a Context that won’t work from the parent process
        # add self.ctx for global, different methods = PUB/SUB
        # self.name = "ZMQAcquirer"

        self.context = zmq.Context.instance()

        self.ip = ip
        self.port = port

        self.multipart = multipart

        self.time_opt = time_opt
        if self.time_opt is True:
            self.timing = timing
            self.out_path = out_path
        
        # win_dur in ms, for example, 20 ms
        # self.seg_dur = (fs*win_dur)/1000.0

        self.done = False

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """
        TODO HERE SYNC SUB w/PUB w/REQ/REP
        https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
        """

        logger.info("Beginning setup for ZMQAcquirer")
        logger.info("Setting up subscriber for acquisition")
        self.setRecvSocket() 

        logger.info("Initializing lists")
        self.in_timestamps = []
        self.zmq_acq_total_times = []
        self.zmq_timestamps = []
        self.get_data = []
        self.put_to_store = []
        self.put_out_time = []

        self.dropped_msg = []
        self.ns = deque([], maxlen=int(100))

        self.seg_num = 0
        self.i = 0

        self.data = deque([], maxlen=int(100))
        # self.data = []

        if self.time_opt is True:
            os.makedirs(self.out_path, exist_ok=True)

        logger.info(f"Completed setup for ZMQAcquirer")


    def stop(self):
        """
        Meh... https://stackoverflow.com/questions/9019873/should-i-close-zeromq-socket-explicitly-in-python
        """
        
        logger.info("ZMQAcquirer stopping")
        # Close subscriber socket
        logger.info("Closing subscriber socket")
        self.recv_socket.close()
        # Terminate context = ONLY terminate if/when BOTH SUB and PUB sockets are closed
        logger.info("Terminating context")
        self.context.term()

        if self.time_opt is True:
            logger.info("Saving out timing info")
            keys = self.timing
            values = [self.in_timestamps, self.zmq_acq_total_times, self.zmq_timestamps, self.get_data, self.put_to_store, self.put_out_time]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.out_path, 'zmq_acq_timing.csv'), index=False, header=True)

        logger.info("ZMQAcquirer stopped")

        return 0
    
    def runStep(self):
        """
        """

        t = time.time()

        if self.time_opt is True:
            self.zmq_timestamps.append(time.time())

        if self.done:
            pass

        try:   
            t1 = time.time()

            logger.info("Receiving msg")
            msg = self.recvMsg()
            logger.info("Msg received: {msg}")

            self.data.extend(np.int16(msg[:-1]))
            self.ns.append(int(msg[-1]))
            logger.info("Data: {self.data}")
            logger.info("ns: {self.ns}")

            print(self.data, '\n', '\n')
            print(self.ns)

            # if np.sum(self.ns) >= int(self.seg_dur) * self.seg_num+1:
            #     t2 = time.time()
            #     self.get_data.append((t2 - t1)*1000.0)
            #     t3 = time.time()
            #     data_obj_id = self.client.put(np.array(self.data), 'seg_num_' + str(self.seg_num))
            #     seg_obj_id = self.client.put(self.seg_num, 'seg_num_' + str(self.seg_num))
            #     t4 = time.time()
            #     self.put_to_store.append((t4 - t3)*1000.0)
            #     t5 = time.time()
            #     self.q_out.put([data_obj_id, seg_obj_id])
            #     self.put_out_time.append((time.time() - t5)*1000.0)

            #     self.seg_num += 1
                
            #     self.zmq_acq_total_times.append((time.time() - t)*1000.0)

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
        self.zmq_acq_total_times.append((time.time() - t)*1000.0)

        print(f"Acquire broke, avg time per segment: {np.mean(self.zmq_acq_total_times)}")
        print(f"Acquire got through {self.seg_num} segments")


    def setRecvSocket(self):
    # def setRecvSocket(self, ip, port):
        """
        ADAPTED CHANG PR ZMQPSActor — limit redundant work

        DO NOT OPEN NEW ZMQ CONTEXT:
        https://stackoverflow.com/questions/45154956/zmq-context-should-i-create-another-context-in-a-new-thread
        https://github.com/zeromq/pyzmq/issues/1172
        https://stackoverflow.com/questions/71312735/inter-process-communication-between-async-and-sync-tasks-using-pyzmq

        TODO: error handling: EINVAL, socket type invalid; EFAULT, context invalid; EMFILE, limit num of open sockets; ETERM, context terminated        

        Sets up the receive socket for the actor — subscriber.
        """

        # self.subscriber = self.context.socket(SUB)
        self.recv_socket = self.context.socket(SUB)

        # connect client node, socket,  with unkown or arbitrary network address(es) to endpoint with well-known network address
        # connect socket to peer address
        # endpoint = peer address:TCP port, source_endpoint:'endpoint'
        # IPv4/IPv6 assigned to interface OR DNS name:TCP port
        recv_address = f"tcp://{self.ip}:{self.port}"
        self.recv_socket.connect(recv_address)
        # receivig messages on all topics
        self.recv_socket.setsockopt(SUBSCRIBE, b"")

        # https://github.com/zeromq/pyzmq/blob/main/examples/pubsub/topics_sub.py
        # for example, timestamps, data
        # b'time: ' or b't' and b'data: ' or b'd'
        # self.recv_socket.setsockopt(SUBSCRIBE, b"t")
        # self.recv_socket.setsockopt(SUBSCRIBE, b"d")

        logger.info(f"Subscriber socket set: {recv_address}")
        # why include a timeout? time how long it takes to connect?
        # time.sleep(timeout)


    def recvMsg(self):
        """
        ADAPTED CHANG PR ZMQPSActor — limit redundant work
        Receives a message from the controller — subscriber.

        DO NOT ADD NOBLOCK HERE
        With flags=NOBLOCK, this raises ZMQError if no messages have arrived
        Will give error: "Resource temporarily unavailable."
        With flags=NOBLOCK, this raises :class:`ZMQError` if no messages have
        arrived; otherwise, this waits until a message arrives.
        See :class:`Poller` for more general non-blocking I/O.
        https://github.com/zeromq/pyzmq/issues/1320
        https://github.com/zeromq/pyzmq/issues/36
        If server given time to receive message, then client when receiving-non-blocking will get message. (BUT WE DO NOT WANT TO WAIT!)
        """

        try:
            if self.multipart:
                msg = self.recv_socket.recv_multipart()
            else: 
                msg = self.recv_socket.recv()
        except ZMQError as e:
            logger.info(f"ZMQ error: {e}")
            if e.errno == ETERM:
                pass           # Interrupted - or break if in loop
            if e.errno == EAGAIN:
                pass  # no message was ready (yet!)
            else:
                raise # real error
                # traceback.print_exc()
        
        logger.info(f"Message received: {msg}")
        return msg # process message
    

if __name__ == "__main__":

    sub_ip = "10.122.168.184"
    sub_port = "5555"

    time_opt = False

    zmq_acq = ZMQAcquirer(name="ZMQAcq", ip=sub_ip, port=sub_port, time_opt=time_opt)

    zmq_acq.setup()
    
    t = time.time()
    while 1:
        zmq_acq.runStep()
        # one = zmq_acq.runZMQAcquirer()
        # print(one - t - 1)