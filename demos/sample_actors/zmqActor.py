import asyncio
import time

import zmq
from zmq import PUB, SUB, SUBSCRIBE, REQ, REP, LINGER, Again, NOBLOCK, ZMQError, EAGAIN, ETERM
from zmq.log.handlers import PUBHandler
import zmq.asyncio

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZmqActor(Actor):
    """
    Zmq actor with pub/sub or rep/req pattern.
    """
    def __init__(self, *args, type='PUB', ip='127.0.0.1', port=5555, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Constructed Zmq Actor")
        if str(type) in 'PUB' or str(type) in 'SUB':
            self.pub_sub_flag = True         #default
        else: self.pub_sub_flag = False
        self.rep_req_flag = not self.pub_sub_flag
        self.ip = ip
        self.port = port
        self.address = "tcp://{}:{}".format(self.ip, self.port)

        self.send_socket = None
        self.recv_socket = None
        self.req_socket = None
        self.rep_socket = None

        self.context = zmq.Context.instance()

    def sendMsg(self, msg, msg_type="pyobj"):
        """
        Sends a message to the controller.
        """
        if not self.send_socket: 
            self.setSendSocket()

        if msg_type == "multipart":
            self.send_socket.send_multipart(msg)
        if msg_type == "pyobj":
            self.send_socket.send_pyobj(msg)
        elif msg_type == "single": 
            self.send_socket.send(msg)

    def recvMsg(self, msg_type="pyobj", flags=0):
        """
        Receives a message from the controller.

        NOTE: default flag=0 instead of flag=NOBLOCK
        """
        if not self.recv_socket: self.setRecvSocket()
        
        while True:
            try:
                if msg_type == "multipart":
                    recv_msg = self.recv_socket.recv_multipart(flags=flags)
                elif msg_type == "pyobj":
                    recv_msg = self.recv_socket.recv_pyobj(flags=flags)
                elif msg_type == "single": 
                    recv_msg = self.recv_socket.recv(flags=flags)
                break
            except Again:
                pass
            except ZMQError as e:
                logger.info(f"ZMQ error: {e}")
                if e.errno == ETERM:
                    pass  # interrupted  - pass or break if in try loop
                if e.errno == EAGAIN:
                    pass  # no message was ready (yet!)

        # self.recv_socket.close()
        return recv_msg

    def requestMsg(self, msg):
        """Safe version of send/receive with controller.
        Based on the Lazy Pirate pattern [here]
        (https://zguide.zeromq.org/docs/chapter4/#Client-Side-Reliability-Lazy-Pirate-Pattern)
        """
        REQUEST_TIMEOUT = 500
        REQUEST_RETRIES = 3
        retries_left = REQUEST_RETRIES

        self.setReqSocket()

        reply = None
        try:
            logger.debug(f"Sending {msg} to controller.")
            self.req_socket.send_pyobj(msg)

            while True:
                ready = self.req_socket.poll(REQUEST_TIMEOUT)

                if ready:
                    reply = self.req_socket.recv_pyobj()
                    logger.debug(f"Received {reply} from controller.")
                    break
                else:
                    retries_left -= 1
                    logger.debug("No response from server.")

                # try to close and reconnect
                self.req_socket.setsockopt(LINGER, 0)
                self.req_socket.close()
                if retries_left == 0:
                    logger.debug("Server seems to be offline. Giving up.")
                    break

                logger.debug("Attempting to reconnect to server...")

                self.setReqSocket()

                logger.debug(f"Resending {msg} to controller.")
                self.req_socket.send_pyobj(msg)

        except asyncio.CancelledError:
            pass

        self.req_socket.close()
        return reply

    def replyMsg(self, reply, delay=0.0001):
        """
        Safe version of receive/reply with controller.
        """

        self.setRepSocket()

        msg = self.rep_socket.recv_pyobj()
        time.sleep(delay)  
        self.rep_socket.send_pyobj(reply)
        self.rep_socket.close()

        return msg

    def put(self, msg=None):
        logger.debug(f'Putting message {msg}')
        if self.pub_sub_flag:
            logger.debug(f"putting message {msg} using pub/sub")
            return self.sendMsg(msg)
        else:
            logger.debug(f"putting message {msg} using rep/req")
            return self.requestMsg(msg)
        
    def get(self, reply=None):
        if self.pub_sub_flag:
            logger.debug(f"getting message with pub/sub")
            return self.recvMsg()
        else:
            logger.debug(f"getting message using reply {reply} with pub/sub")
            return self.replyMsg(reply)
    
    def setSendSocket(self, timeout=0.001):
        """
        Sets up the send socket for the actor.
        """
        self.send_socket = self.context.socket(PUB)
        self.send_socket.bind(self.address)
        time.sleep(timeout)

    def setRecvSocket(self, timeout=0.001):
        """
        Sets up the receive socket for the actor.
        """
        self.recv_socket = self.context.socket(SUB)
        self.recv_socket.connect(self.address)
        self.recv_socket.setsockopt(SUBSCRIBE, b"")
        time.sleep(timeout)
    
    def setReqSocket(self, timeout=0.0001):
        """
        Sets up the request socket for the actor.
        """
        self.req_socket = self.context.socket(REQ)
        self.req_socket.connect(self.address)
        time.sleep(timeout)

    def setRepSocket(self, timeout=0.0001):
        """
        Sets up the reply socket for the actor.
        """
        self.rep_socket = self.context.socket(REP)
        self.rep_socket.bind(self.address)
        time.sleep(timeout)
