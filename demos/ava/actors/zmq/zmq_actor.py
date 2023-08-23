# import asyncio
import time
import traceback

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, ETERM, EAGAIN
from zmq.log.handlers import PUBHandler

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZMQActor(Actor):
    """Base class for an actor that uses ZMQ.

    TODO:
    - ONE ACTOR, ONE CONTEXT, TWO DIFF SOCKETS 
    - MULTIPLE ACTORS (ACQ AND STIM) = SHARED CONTEXT, TWO DIFF SOCKETS
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        logger.info("Instantiating a global ZMQ context.")
        self.context = zmq.Context.instance()


    def setRecvSocket(self, ip, port, timeout=0.001):
        """_summary_

        Args:
            ip (_type_): _description_
            port (_type_): _description_
        """
        self.recv_socket = self.context.socket(SUB)
        recv_address = f"tcp://{ip}:{port}"
        self.recv_socket.connect(recv_address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)

        logger.info(f"Subscriber socket set: {recv_address}")

    def recvMsg(self, msg_type="multipart"):
        """
        """

        try:
            if msg_type == "multipart":
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
        
        # logger.info(f"Message received: {msg}")
        return msg # process message