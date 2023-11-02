import numpy as np
import os
import pandas as pd
import time

from queue import Empty

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, ETERM, EAGAIN, RCVBUF, RCVHWM
# from actors.zmq.zmq_actor import ZMQActor

from improv.actor import Actor
from improv.store import CannotGetObjectError, ObjectNotFoundError

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ZMQPublisher(Actor):
    """Actor for acquiring audio as ZMQ messages sent from LabVIEW-EvTAF18 sent over TCP to improv.
    """

    def __init__(self, *args, port=None, time_opt=True, timing=None, out_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.port = port

        self.time_opt = time_opt
        if self.time_opt:
            self.timing = timing
            self.timing_path = os.path.join(out_path, 'timing')
            logger.info(f"{self.name} timing directory: {self.timing_path}")

        self.done = False

        self.context = zmq.Context.instance()

        self.send_address = f"tcp://*:{port}"

    def __str__(self):
        return f"Name: {self.name}, Data: {self.data}"

    def setup(self):
        """_summary_
        """

        logger.info(f"Running setup for {self.name}.")

        self.send_socket = self.context.socket(PUB)
        self.send_socket.bind(self.send_address)
        logger.info(f"Publisher socket set: {self.send_address}")
        
        self.zmq_send_total_times = []

        if self.time_opt:
            os.makedirs(self.timing_path, exist_ok=True)

            logger.info(f"Initializing lists for {self.name} timing.")
            
            self.zmq_send_timestamps = []
            self.send_msgs = []
            self.timestamps = []

        self.msg_nums = []
        self.msg_num = 0

        logger.info(f"Completed setup for {self.name}.")
    
    def stop(self):
        
        logger.info(f"{self.name} stopping.")

        logger.info(f"Publisher avg time per message: {np.mean(self.zmq_send_total_times)}")
        logger.info(f"Publisher got through {self.msg_num} messages.")

        # Close subscriber socket
        self.send_socket.close()

        if self.time_opt:
            logger.info(f"Saving timing info for {self.name}.")
            keys = self.timing
            values = [self.zmq_send_timestamps, self.send_msgs, self.zmq_send_total_times, self.msg_nums]

            timing_dict = dict(zip(keys, values))
            df = pd.DataFrame.from_dict(timing_dict, orient='index').transpose()
            df.to_csv(os.path.join(self.timing_path, 'pub_timing_' + str(self.msg_num) + '.csv'), index=False, header=True)
        
        logger.info(f"{self.name} stopped.")
        
        return 0
    
    def runStep(self):
        """_summary_
        """

        if self.done:
            pass

        t = time.perf_counter_ns()

        if self.time_opt:
            self.zmq_send_timestamps.append(t)

        ids = self._checkInput()

        if ids is not None:
            t = time.perf_counter_ns()
            self.done = False
            try:
                # msg = self.client.getID(ids)
                msg_timestamp = self.client.getID(ids[0])

                if int(self.msg_num) % 2 == 0:
                    msg = 'true'
                elif int(self.msg_num) % 2 != 0:
                    msg = 'false'

                self.send_socket.send_string(str(self.msg_num), zmq.SNDMORE)
                self.send_socket.send_string(str(msg), zmq.SNDMORE)
                self.send_socket.send_string(str(msg_timestamp), zmq.SNDMORE)
                self.send_socket.send_string(str(time.perf_counter_ns()))
                t_send = time.perf_counter_ns()

                self.msg_nums.append(int(self.msg_num))
                self.msg_num += 1

                if self.time_opt:
                    self.send_msgs.append(t_send)

            except Exception as e:
                logger.error(f"{self.name} general exception: {e}")
            # except IndexError as e:
            #     pass

            except ObjectNotFoundError:
                logger.error(f"{self.name}: Message {self.seg_num} unavailable from store, dropping")
                # self.dropped_wav.append(self.seg_num)
            except KeyError as e:
                logger.error(f"{self.name}: Key error... {e}")
                # self.dropped_wav.append(self.seg_num)
            except Exception as e:
                logger.error(f"{self.name} error: {type(e).__name__}: {e} during segment number {self.msg_num}")
                logger.info(traceback.format_exc())
                # self.dropped_wav.append(self.seg_num)

            self.zmq_send_total_times.append((time.perf_counter_ns() - t) * 10**-3)

    def _checkInput(self):
        ''' Check to see if we have .wav — q_in
        '''
        try:
            res = self.q_in.get(timeout=0.005)
            return res
        #TODO: additional error handling
        except Empty:
            pass
            # logger.info('No .wav files for processing')
            # return None
